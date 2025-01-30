''' Generate images and shapes using pretrained network pickle.
Code adapted from following paper
"Efficient Geometry-aware 3D Generative Adversarial Networks."
See LICENSES/LICENSE_EG3D for original license.
'''

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import math
import PIL.Image
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import mrcfile
import trimesh
import mcubes
import json

import legacy
from camera_utils import FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------
def process_img(img):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img

#----------------------------------------------------------------------------
class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, lookat_position, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        forward_vectors = normalize_vecs(lookat_position - camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world
#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--latent', type=str, help='latent code', required=False)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=40, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=True, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=25, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--pose_cond', type=int, help='camera conditioned pose angle', default=90, show_default=True)

def generate_images(
    network_pkl: str,
    latent: str,
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    pose_cond: int,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained model.
    python gen_samples.py --outdir=out --trunc=0.7 --shapes=true --seeds=0-3 \
        --network models/easy-khair-180-gpc0.8-trans10-025000.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda:0')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    reload_modules = True

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
    # reset the near plane for better recrop
    G.rendering_kwargs["ray_start"] = 2.2
    
    network_name = os.path.basename(network_pkl)
    # mesh_id = str(network_pkl).split('/')[-2]
    outdir_rgb = os.path.join(outdir, 'rgb')
    # outdir_depth = os.path.join(outdir, mesh_id, 'depth')
    outdir_normal = os.path.join(outdir, 'normal')
    outdir_mask = os.path.join(outdir, 'mask')
    os.makedirs(outdir_rgb, exist_ok=True)
    # os.makedirs(outdir_depth, exist_ok=True)
    os.makedirs(outdir_normal, exist_ok=True)
    os.makedirs(outdir_mask, exist_ok=True)

    json_path = os.path.join(outdir, 'cam_params.json')

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    cam_pivot = torch.tensor([0, 0, 0], device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 3.0)

    ws = torch.tensor(np.load(latent)['w']).to(device)

    imgs = []
    normal_imgs = []
    depth_imgs = []
    masks = []

    angle_y_samples = np.arange(0, 360, 360/40)

    angle_p_samples = np.arange(-5, -6, -1)

    samples = [(np.deg2rad(angle_y), np.deg2rad(angle_p)) for angle_y in angle_y_samples for angle_p in angle_p_samples]
    cam_params_dict = {}
    for idx, angles in enumerate(samples):

        angle_y = angles[0]
        angle_p = angles[1]

        # rand camera setting
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        cam_params_dict[idx] = camera_params.tolist()

        out = G.synthesis(ws, camera_params, neural_rendering_resolution=300)
        img = out["image"]
        normal_image = out["image_normal"]
        # depth_img = out["image_depth"]
        mask = out["image_mask"]
        target_size = (img.shape[2], img.shape[3])

        normal_image = (((normal_image.permute(0, 2, 3, 1) + 1.0) / 2.0) * 255).to(torch.uint8)

        img = process_img(img)

        # depth_img -= depth_img.min()
        # depth_img /= depth_img.max()
        # depth_img -= .5
        # depth_img *= -2
        # depth_img = process_img(depth_img)

        mask = (mask.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
        mask[mask < 250] = 0

        mask_img = mask.squeeze().detach().cpu().numpy()

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_img, connectivity=8)

        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

        filled_mask = np.zeros_like(mask_img)
        filled_mask[labels == largest_label] = 255


        filled_mask = np.expand_dims(filled_mask, axis=-1)

        masked_normal_image = normal_image.squeeze().cpu().numpy() * (filled_mask // 255) + (1-(filled_mask // 255))*255

        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir_rgb}/{idx:03d}.png')
        PIL.Image.fromarray(masked_normal_image, 'RGB').save(f'{outdir_normal}/{idx:03d}.png')
        # PIL.Image.fromarray(depth_img[0][:, :, 0].cpu().numpy(), 'L').save(f'{outdir_depth}/{idx:03d}.png')
        PIL.Image.fromarray(filled_mask[:, :, 0], 'L').save(f'{outdir_mask}/{idx:03d}.png')

    with open(json_path, 'w') as json_file:
        json.dump(cam_params_dict, json_file)

    if shapes:
        # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
        ws = torch.tensor(np.load(latent)['w']).to(device)

        max_batch=1000000

        samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
        samples = samples.to(ws.device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=ws.device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=ws.device)
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        with tqdm(total = samples.shape[1]) as pbar:
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    # sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                    sigma = G.sample_mixed(samples[:, head:head + max_batch],
                                     transformed_ray_directions_expanded[:, :samples.shape[1] - head], ws,
                                     truncation_psi=truncation_psi,
                                     truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                    sigmas[:, head:head+max_batch] = sigma
                    head += max_batch
                    pbar.update(max_batch)

        sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
        sigmas = np.flip(sigmas, 0)
        # vtx, faces = mcubes.marching_cubes(sigmas, 1.0)
        # vtx = vtx / (sigmas.shape[0] - 1) * 2
        # vtx_ray_directions_expanded = torch.zeros((vtx.shape[0], vtx.shape[1], 3), device=ws.device)
        # vtx_ray_directions_expanded[..., -1] = -1
        # import pdb
        # pdb.set_trace()
        # vtx_tensor = torch.tensor(vtx, dtype=torch.float32, device=ws.device).unsqueeze(0)
        # # vtx_colors = self.model.synthesizer.forward_points(planes, vtx_tensor) ['rab'].squeeze(0).cpu().numpy()  # (0, 1)
        #
        # vtx_colors = G.sample_rgb(vtx_tensor, vtx_ray_directions_expanded, ws,
        #                     truncation_psi=truncation_psi,
        #                     truncation_cutoff=truncation_cutoff, noise_mode='const')
        #
        #
        # # from training.volumetric_rendering.ray_marcher import MipRayMarcher2
        #
        # vtx_colors = (vtx_colors * 255).astype(np.uint8)
        # mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)

        # Trim the border of the extracted cube
        pad = int(30 * shape_res / 256)
        pad_value = -1000
        sigmas[:pad] = pad_value
        sigmas[-pad:] = pad_value
        sigmas[:, :pad] = pad_value
        sigmas[:, -pad:] = pad_value
        sigmas[:, :, :pad] = pad_value
        sigmas[:, :, -pad:] = pad_value
        seed = 2


        if shape_format == '.ply':
            from shape_utils import convert_sdf_samples_to_ply
            convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'init_mesh.ply'), level=20)
        # elif shape_format == '.mrc': # output mrc
        #     filepath = os.path.join(outdir, f'id_{mesh_id}.mrc')
        #     print(f'writing mesh in {filepath}')
        #     with mrcfile.new_mmap(filepath, overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
        #         mrc.data[:] = sigmas
            print(outdir)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
