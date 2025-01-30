import bisect
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
import trimesh
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
import json


import threestudio
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *


@dataclass
class SingleImageDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 96
    width: Any = 96
    # resolution_milestones: List[int] = field(default_factory=lambda: [])
    default_elevation_deg: float = 0.0
    default_azimuth_deg: float = -180.0
    default_camera_distance: float = 1.2
    default_fovy_deg: float = 60.0
    image_path: str = ""
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 2e-3
    batch_size: int = 1
    requires_depth: bool = False
    requires_normal: bool = False
    rays_d_normalize: bool = True
    use_mixed_camera_config: bool = False


class SingleImageDataBase:
    def setup(self, cfg, split):

        self.split = split
        self.rank = get_rank()
        self.cfg: SingleImageDataModuleConfig = cfg

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            # FIXME:
            if self.cfg.use_mixed_camera_config:
                if self.rank % 2 == 0:
                    random_camera_cfg.camera_distance_range=[self.cfg.default_camera_distance, self.cfg.default_camera_distance]
                    random_camera_cfg.fovy_range=[self.cfg.default_fovy_deg, self.cfg.default_fovy_deg]
                    self.fixed_camera_intrinsic = True
                else:
                    self.fixed_camera_intrinsic = False
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]

        self.load_images()

        mesh_path = os.path.join(self.root_dir, 'init_mesh.ply')
        mesh = trimesh.load(mesh_path)
        # move to center
        centroid = mesh.vertices.mean(0)
        mesh.vertices = mesh.vertices - centroid

        scale = np.abs(mesh.vertices).max()

        z_, x_ = (
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
        )
        y_ = np.cross(z_, x_)
        std2mesh = np.stack([x_, y_, z_], axis=0).T
        mesh2std = np.linalg.inv(std2mesh)

        # shift = np.mean(vertices, axis=0)
        # scale = np.max(np.linalg.norm(vertices - shift, ord=2, axis=1))

        landmark_path = os.path.join(self.root_dir, 'laion.txt')
        with open(landmark_path, "r") as file:
            lines = file.readlines()
        coordinates = [list(map(float, line.strip().split())) for line in lines]
        face_landmarks = np.array(coordinates)
        self.face_landmarks = (face_landmarks - centroid) / scale * 0.9
        # import pdb
        # pdb.set_trace()
        self.face_landmarks = np.dot(mesh2std, self.face_landmarks.T).T
        self.face_landmarks = torch.from_numpy(self.face_landmarks).to(torch.float32)

        camera = np.load(os.path.join(self.root_dir, 'camera.npy'))
        cam2world_matrix = camera[:16].reshape((4, 4))
        intrinsics = camera[16:25].reshape((3, 3))

        cam_position = torch.tensor(cam2world_matrix[:3, 3])
        x, y, z = cam_position[0], cam_position[1], cam_position[2]

        # align to up-z and front-x
        camera_distance = torch.norm(cam_position)  # 2.7
        # 计算仰角
        elevation = torch.asin(y / camera_distance)
        # 计算方位角
        azimuth = torch.atan2(z, x)

        elevation_deg = torch.FloatTensor([elevation])
        azimuth_deg = torch.FloatTensor([azimuth])

        camera_position = (cam_position - centroid) /scale * 0.9
        camera_position = np.dot(mesh2std, camera_position.T).T
        # import pdb
        # pdb.set_trace()
        camera_position = torch.from_numpy(camera_position).unsqueeze(0).to(torch.float32)

        center: Float[Tensor, "1 3"] = torch.zeros_like(camera_position)
        centroid = torch.tensor(centroid)
        center -= centroid
        up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

        light_position: Float[Tensor, "1 3"] = camera_position
        lookat: Float[Tensor, "1 3"] = F.normalize(center - camera_position, dim=-1)
        right: Float[Tensor, "1 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)

        rotate_matrix = np.array([
                        [0, 0, 1],
                        [1, 0, 0],
                        [0, 1, 0]
                    ])
        rotated_cam = np.dot(rotate_matrix, cam2world_matrix[:3, :4])
        self.c2w = torch.from_numpy(rotated_cam).unsqueeze(0).to(torch.float32)
        # self.c2w = torch.tensor(cam2world_matrix[:3, :4]).unsqueeze(0).to(torch.float32)
        self.c2w[:, :, 3] = camera_position

        self.c2w4x4: Float[Tensor, "B 4 4"] = torch.cat(
            [self.c2w, torch.zeros_like(self.c2w[:, :1])], dim=1
        )
        self.c2w4x4[:, 3, 3] = 1.0

        self.camera_position = camera_position
        self.light_position = light_position
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distance = camera_distance
        # self.fovy = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg]))
        self.fovy = torch.tensor([2 * np.arctan(1 / (2 * 4.2647))])

        # self.resolution_milestones: List[int]
        # if len(self.heights) == 1 and len(self.widths) == 1:
        #     if len(self.cfg.resolution_milestones) > 0:
        #         threestudio.warn(
        #             "Ignoring resolution_milestones since height and width are not changing"
        #         )
        #     self.resolution_milestones = [-1]
        # else:
        #     assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
        #     self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.focal_lengths = [
            0.5 * height / torch.tan(0.5 * self.fovy) for height in self.heights
        ]

        self.directions_unit_focal = self.directions_unit_focals[0]
        self.focal_length = self.focal_lengths[0]
        self.set_rays()

        self.prev_height = self.height

    def set_rays(self):
        # get directions by dividing directions_unit_focal by focal length
        directions: Float[Tensor, "1 H W 3"] = self.directions_unit_focal[None]
        directions[:, :, :, :2] = directions[:, :, :, :2] / self.focal_length

        rays_o, rays_d = get_rays(
            directions,
            self.c2w,
            keepdim=True,
            noise_scale=2e-3,
            normalize=True,
        )

        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.01, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(self.c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx

    def load_images(self):
        # load image
        self.root_dir = self.cfg.image_path
        if self.root_dir.endswith('.png'):
            self.root_dir = os.path.dirname(self.root_dir)
        image_path = os.path.join(self.root_dir, 'img.png')
        normal_path = os.path.join(self.root_dir, 'img_normal.png')
        depth_path = os.path.join(self.root_dir, 'img_depth.png')
        mask_path = os.path.join(self.root_dir, 'face_mask.png')
        # control_img_path = os.path.join(self.root_dir, 'img_canny.png')

        assert os.path.exists(image_path), f"Could not find image {self.cfg.image_path}!"

        rgb = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        rgb = (
            cv2.resize(
                rgb, (self.width, self.height), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            / 255.0
        )
        self.rgb: Float[Tensor, "1 H W 3"] = (
            torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank)
        )

        assert os.path.exists(normal_path)
        normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
        normal = cv2.resize(
            normal, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        self.normal: Float[Tensor, "1 H W 3"] = (
            torch.from_numpy(normal.astype(np.float32) / 255.0)
            .unsqueeze(0).contiguous()
            .to(self.rank)
        )

        assert os.path.exists(depth_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(
            depth, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        self.depth: Float[Tensor, "1 H W 3"] = (
            torch.from_numpy(depth.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .to(self.rank)
        )

        assert os.path.exists(mask_path), f"Could not find mask image {self.cfg.image_path}!"

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.mask: Float[Tensor, "1 H W 1"] = (
            torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).contiguous().to(self.rank)
        ).unsqueeze(-1)
        self.rgb =self.rgb * self.mask
        self.normal = self.normal * self.mask

        self.depth_for_mask = self.depth.permute(0, 3, 1, 2)[:,0,:,:].unsqueeze(0)
        # self.depth = self.depth * self.mask
        mask = (self.mask > 0).permute(0, 3, 1, 2)
        mask_minmax = self.depth_for_mask > 0.2
        # import pdb; pdb.set_trace()
        max_depth = torch.max(self.depth_for_mask[mask_minmax])
        min_depth = torch.min(self.depth_for_mask[mask_minmax])
        self.depth = torch.lerp(
                torch.zeros_like(self.depth_for_mask), (self.depth_for_mask - min_depth) / (max_depth - min_depth + 1e-7), mask_minmax.float()
            )
        # import pdb;pdb.set_trace()
        # control_img_ref =  cv2.imread(control_img_path, cv2.IMREAD_UNCHANGED)
        # self.control_img_ref = cv2.resize(control_img_ref, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # self.control_img_ref = torch.from_numpy(self.control_img_ref).float().to(self.rank) / 255.0
        # self.control_img_ref = self.control_img_ref.unsqueeze(0)
        # self.control_img_ref = self.control_img_ref.permute(0, 3, 1, 2)

        print(
            f"[INFO] single image dataset: load image {self.cfg.image_path} {self.rgb.shape}"
        )

    def get_all_images(self):
        return self.rgb


class SingleImageIterableDataset(IterableDataset, SingleImageDataBase, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def collate(self, batch) -> Dict[str, Any]:
        batch = {
            "rays_o": self.rays_o.squeeze(0),
            "rays_d": self.rays_d.squeeze(0),
            "mvp_mtx": self.mvp_mtx.squeeze(0),
            "camera_positions": self.camera_position.squeeze(0),
            "light_positions": self.light_position.squeeze(0),
            "elevation": self.elevation_deg,
            "azimuth": self.azimuth_deg,
            "camera_distances": self.camera_distance,
            "rgb": self.rgb,
            "mask": self.mask.squeeze(0),
            "depth_for_mask": self.depth_for_mask.squeeze(0),
            "depth": self.depth.squeeze(0),
            "normal": self.normal.squeeze(0),
            # "ref_control_img": self.control_img_ref.squeeze(0),
            "mask": self.mask,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "c2w": self.c2w.squeeze(0),
            "c2w4x4": self.c2w4x4.squeeze(0),
            "face_landmarks": self.face_landmarks,
        }

        return batch

    def __iter__(self):
        while True:
            yield {}


class SingleImageDataset(Dataset, SingleImageDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def __len__(self):
        return len(self.random_pose_generator)

    def __getitem__(self, index):
        batch = self.random_pose_generator[index]

        batch.update(
            {
            "height": self.random_pose_generator.cfg.eval_height,
            "width": self.random_pose_generator.cfg.eval_width,
            "mvp_mtx_ref": self.mvp_mtx[0],
            "c2w_ref": self.c2w4x4,
            }
        )
        return batch


class MultiImagesDataset(Dataset):
    def __init__(self, cfg, split='train'):

        self.cfg = cfg
        self.rank = get_rank()
        self.root_dir = self.cfg.image_path
        self.image_dir = os.path.join(self.root_dir, 'rgb_vis')
        self.normal_dir = os.path.join(self.root_dir, 'normal_vis')
        self.mask_dir = os.path.join(self.root_dir, 'mask')
        self.cam_file_path = os.path.join(self.root_dir, 'cam_params.json')

        assert os.path.exists(self.image_dir), "Image directory does not exist"
        assert os.path.exists(self.normal_dir), "Normal directory does not exist"
        assert os.path.exists(self.mask_dir), "Mask directory does not exist"
        assert os.path.exists(self.cam_file_path), "Camera parameters file does not exist"

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            # FIXME:
            if self.cfg.use_mixed_camera_config:
                if self.rank % 2 == 0:
                    random_camera_cfg.camera_distance_range=[self.cfg.default_camera_distance, self.cfg.default_camera_distance]
                    random_camera_cfg.fovy_range=[self.cfg.default_fovy_deg, self.cfg.default_fovy_deg]
                    self.fixed_camera_intrinsic = True
                else:
                    self.fixed_camera_intrinsic = False
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )
        self.ref_img_generator = SingleImageIterableDataset(self.cfg, split)

        # cal mesh scale
        mesh_path = os.path.join(self.root_dir, 'init_mesh.ply')
        mesh = trimesh.load(mesh_path)
        # move to center
        centroid = mesh.vertices.mean(0)
        mesh.vertices = mesh.vertices - centroid

        scale = np.abs(mesh.vertices).max()

        z_, x_ = (
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
        )
        y_ = np.cross(z_, x_)
        std2mesh = np.stack([x_, y_, z_], axis=0).T
        mesh2std = np.linalg.inv(std2mesh)

        # move face landmark
        landmark_path = os.path.join(self.root_dir, 'laion.txt')
        with open(landmark_path, "r") as file:
            lines = file.readlines()
        coordinates = [list(map(float, line.strip().split())) for line in lines]
        face_landmarks = np.array(coordinates)
        self.face_landmarks = (face_landmarks - centroid) / scale * 0.9

        self.face_landmarks = np.dot(mesh2std, self.face_landmarks.T).T
        self.face_landmarks = torch.from_numpy(self.face_landmarks).to(torch.float32)

        with open(self.cam_file_path, 'r') as f:
            self.cam_data = json.load(f)

        self.image_list = sorted(os.listdir(self.image_dir))

        cam_params = torch.tensor(self.cam_data[str(1)])
        intrinsics = cam_params[0, 16:25].reshape((3, 3))
        self.fovy = torch.tensor([2 * np.arctan(1 / (2 * intrinsics[0, 0]))])  # 与panohead中计算方法保持一致

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]

        self.focal_lengths = [
            0.5 * height / torch.tan(0.5 * self.fovy) for height in self.heights
        ]
        self.focal_length = self.focal_lengths[0]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]

        self.directions_unit_focal = self.directions_unit_focals[0]

        self.params_list = []
        for idx in range(len(self.image_list)):
            cam_params = torch.tensor(self.cam_data[str(idx)])
            cam2world_matrix = cam_params[0, :16].reshape((4, 4))
            cam2world_matrix[:, 1:3] = -cam2world_matrix[:, 1:3]
            intrinsics = cam_params[0, 16:25].reshape((3, 3))
            cam_position = torch.tensor(cam2world_matrix[:3, 3])
            x, y, z = cam_position[0], cam_position[1], cam_position[2]
            # align to up-z and front-x
            camera_distance = torch.norm(cam_position)  # 2.7

            elevation = torch.asin(y / camera_distance)
            azimuth = torch.atan2(z, x)

            elevation_deg = torch.rad2deg(torch.FloatTensor([elevation]))
            azimuth_deg = torch.rad2deg(torch.FloatTensor([azimuth])) - 90.0

            camera_position = (cam_position - centroid) / scale * 0.9
            camera_position = np.dot(mesh2std, camera_position.T).T

            camera_position = torch.from_numpy(camera_position).unsqueeze(0).to(torch.float32)
            camera_distance = torch.norm(cam_position)

            light_position = camera_position

            rotate_matrix = np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ])
            rotated_cam = np.dot(rotate_matrix, cam2world_matrix[:3, :4])
            c2w = torch.from_numpy(rotated_cam).unsqueeze(0).to(torch.float32)
            # self.c2w = torch.tensor(cam2world_matrix[:3, :4]).unsqueeze(0).to(torch.float32)
            c2w[:, :, 3] = camera_position

            c2w4x4: Float[Tensor, "B 4 4"] = torch.cat(
                [c2w, torch.zeros_like(c2w[:, :1])], dim=1
            )
            c2w4x4[:, 3, 3] = 1.0


            proj_mtx = get_projection_matrix(
                self.fovy, self.width / self.height, 0.01, 100.0
            )

            mvp_mtx = get_mvp_matrix(c2w, proj_mtx)

            rays_o, rays_d = self.set_rays(c2w)

            params = {
                "cam_position": cam_position,
                "light_position": light_position,
                "elevation": elevation_deg,
                "azimuth": azimuth_deg,
                "camera_distance": camera_distance,
                "c2w": c2w.squeeze(),
                "c2w4x4": c2w4x4.squeeze(),
                "mvp_mtx": mvp_mtx.squeeze(),
                "rays_o": rays_o.squeeze(),
                "rays_d": rays_d.squeeze(),
            }

            self.params_list.append(params)

    def set_rays(self, c2w):
        # get directions by dividing directions_unit_focal by focal length
        directions: Float[Tensor, "1 H W 3"] = self.directions_unit_focal[None]
        directions[:, :, :, :2] = directions[:, :, :, :2] / self.focal_length


        rays_o, rays_d = get_rays(
            directions,
            c2w,
            keepdim=True,
            noise_scale=2e-3,
            normalize=True,
        )

        return rays_o, rays_d

    def compute_elevation_azimuth_from_position(self, cam_position):
        x, y, z = cam_position[0], cam_position[1], cam_position[2]

        elevation = torch.asin(z / torch.norm(cam_position))

        azimuth = torch.atan2(y, x)

        elevation_deg = elevation * math.pi / 180
        azimuth_deg = azimuth * math.pi / 180

        return elevation_deg, azimuth_deg

    def __len__(self):
        return self.cfg.max_steps

    def __getitem__(self, idx):
        idx = idx % (len(self.image_list))
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        normal_path = os.path.join(self.normal_dir, img_name)
        # depth_path = os.path.join(self.depth_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # load rgb
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(
            image, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        image = torch.from_numpy(image.astype(np.float32) / 255.0).to(self.rank)

        # load normal
        normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
        normal = cv2.resize(
            normal, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        normal = torch.from_numpy(normal.astype(np.float32) / 255.0).to(self.rank)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_AREA)
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).to(self.rank)

        # Get camera parameters
        params = self.params_list[idx]

        # Assuming you want to return a dictionary with the required components
        sample = {
            "rays_o": params['rays_o'],
            "rays_d":  params['rays_d'],
            'mvp_mtx': params["mvp_mtx"],
            "camera_positions": params["cam_position"],
            "light_positions": params["light_position"],
            "elevation": params["elevation"],
            "azimuth": params["azimuth"],
            "camera_distances": params["camera_distance"],
            'rgb': image,
            'ref_normal': normal,
            'mask': mask,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "c2w": params["c2w"],
            "c2w4x4": params["c2w4x4"],
            "face_landmarks": self.face_landmarks,
        }

        sample["ref_control_img"] = self.ref_img_generator.collate(None)

        if self.cfg.use_random_camera:
            sample["random_camera"] = self.random_pose_generator.collate(None)
            sample["random_camera"]['rays_o'] = sample["random_camera"]['rays_o'].squeeze(0)
            sample["random_camera"]['rays_d'] = sample["random_camera"]['rays_d'].squeeze(0)
            sample["random_camera"]['mvp_mtx'] = sample["random_camera"]['mvp_mtx'].squeeze(0)
            sample["random_camera"]['camera_positions'] = sample["random_camera"]['camera_positions'].squeeze(0)
            sample["random_camera"]['c2w'] = sample["random_camera"]['c2w'].squeeze(0)
            sample["random_camera"]['proj_mtx'] = sample["random_camera"]['proj_mtx'].squeeze(0)

        return sample


@register("coarse2fine-datamodule")
class ImagesDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = MultiImagesDataset(self.cfg, split='train')
        if stage in [None, "fit", "validate"]:
            self.val_dataset = SingleImageDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = SingleImageDataset(self.cfg, "test")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=1, shuffle=True, num_workers=0
        )
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=1, shuffle=False, num_workers=0
        )
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, num_workers=0
        )


if __name__=='__main__':
    from threestudio.utils.config import ExperimentConfig, load_config
    import yaml
    from easydict import EasyDict

    # parse YAML config to OmegaConf
    config_path = 'path to geomrtry config'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)

    dataset = MultiImagesDataset(cfg.data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        # Access the elements in the batch
        rays_o = batch['rays_o']
        rays_d = batch['rays_d']
        mvp_mtx = batch['mvp_mtx']

        # Print or inspect the contents of the batch
        print("Batch Size:", rays_o.size(0))
        print("Rays Origin Shape:", rays_o.shape)
        print("MVP Matrix Shape:", mvp_mtx.shape)
        import pdb
        pdb.set_trace()

        # Break the loop after processing one batch

