import os
import pickle
import json
import torch
import numpy as np
from plyfile import PlyData, PlyElement
import argparse
from tqdm import tqdm

from common.cor.correspondences import Correspondences
from common.cor.octree import OCTree
from common.mesh import TriMesh
from common.transform.pose_estimate import PoseEstimate


def translation_and_scale(source, target):
    u_s = np.mean(source, axis=0)
    u_t = np.mean(target, axis=0)

    sigma_s = np.mean(np.sum((source - u_s) ** 2, axis=1))
    sigma_t = np.mean(np.sum((target - u_t) ** 2, axis=1))

    s = np.sqrt(sigma_t / sigma_s)

    t = u_t - s * u_s

    scaled_source = s * source

    transformed_source = scaled_source + t

    return s, t


def transform_point_cloud(point_cloud, R, S, T):
    point_cloud_np = np.array(point_cloud)

    ones_column = np.ones((point_cloud_np.shape[0], 1))
    point_cloud_np = np.hstack((point_cloud_np, ones_column))

    transformation_matrix = T @ S @ R

    transformed_point_cloud = (transformation_matrix @ point_cloud_np.T).T

    return transformed_point_cloud[:, :3]


class LdmkProjector:
    def __init__(self):
        self.cor_builder = Correspondences()

    def get_kps_coor(self, cam2world_matrix, intrinsics, kps):
        cam2world_matrix = torch.tensor(cam2world_matrix, dtype=torch.float32).unsqueeze(0)
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        sk = intrinsics[0, 1]
        # normalized_keypoints = kps / 564.0
        normalized_keypoints = kps / 1126.0
        keypoints = torch.tensor(normalized_keypoints, dtype=torch.float32)

        # coords = np.array([(i, j) for i in range(100) for j in range(100)])
        # coords = np.hstack((coords[:, 0].reshape(-1, 1), coords[:, 1].reshape(-1, 1)))
        # img_coords = torch.tensor(coords / 100.0, dtype=torch.float32)
        # import pdb
        # pdb.set_trace()
        # x_cam = img_coords[:, 0].view(1, -1)
        # y_cam = img_coords[:, 1].view(1, -1)
        # z_cam = torch.ones((x_cam.shape[0], x_cam.shape[1]), device=keypoints.device)
        # x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1) - sk.unsqueeze(
        #     -1) * y_cam / fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
        # y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam
        # cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)
        # world_image_points = torch.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]
        # import pdb
        # pdb.set_trace()
        # transform_dense = world_image_points.squeeze(0).numpy()
        # np.savetxt(os.path.join('process_data', 'image.txt'), transform_dense, fmt='%f %f %f')

        x_cam = keypoints[:, 0].view(1, -1)
        y_cam = keypoints[:, 1].view(1, -1)
        z_cam = torch.ones((x_cam.shape[0], x_cam.shape[1]), device=keypoints.device)
        x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1) - sk.unsqueeze(
            -1) * y_cam / fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam
        
        cam_rel_points = torch.stack((x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)

        world_rel_points = torch.bmm(cam2world_matrix, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]

        return world_rel_points.squeeze()

    def inverse_project(self, scan, scan_octree, lmk, cam_mat, trans_mat):

        lmk_3d_norm = self.get_kps_coor(trans_mat, cam_mat, lmk)
        
        tm_inv = np.linalg.inv(trans_mat)
        
        # ray_start = np.tile(tm_inv[:3, 3], lmk_3d_norm.shape[0]).reshape((-1, 3))

        ray_start = np.tile(cam2world_matrix[:3, 3], (lmk_3d_norm.shape[0], 1))

        ray_dir = lmk_3d_norm.numpy() - ray_start
        ray_dir = ray_dir / (np.sqrt(np.sum(ray_dir ** 2, axis=1))[:, np.newaxis] + 1e-12)


        step_size = 0.01

        ray_mesh = TriMesh()
        ray_mesh.vertices = ray_start
        # ray_mesh.vert_normal = -1.0* ray_dir
        ray_mesh.vert_normal = ray_dir


        ray_ind, tgt_face_ind, weights = self.cor_builder.nearest_tri_normal(ray_mesh, scan, scan_octree,
                                                                             dist_threshold=np.inf - 10,
                                                                             normal_threshold=-1)
        del ray_mesh
        return ray_ind, tgt_face_ind, weights

    def ldmk_3d_detect(self, mesh, lmk2d, cam_mat, trans_mat):

        # vertices = np.vstack(mesh['vertex']['x'], mesh['vertex']['y'], mesh['vertex']['z']).T
        vertices = np.concatenate([mesh['vertex']['x'].reshape(-1, 1),
                                   mesh['vertex']['y'].reshape(-1, 1),
                                   mesh['vertex']['z'].reshape(-1, 1)], axis=1)
        face = np.vstack(mesh['face']['vertex_indices'])
        scan = TriMesh(vertices=vertices, faces=face)
        positive_z_indices = np.where(vertices[:, 2] > 0)[0]
        scan.del_by_vert_ind(positive_z_indices)

        octree = OCTree()
        octree.from_triangles(scan.vertices, scan.faces, np.arange(scan.face_num()))

        src_ind, tgt_face_ind, weights = self.inverse_project(scan, octree, lmk2d, cam_mat, trans_mat)

        lmk3d_lst = np.sum(scan.vertices[scan.faces[tgt_face_ind]] * weights[:, :, np.newaxis], axis = 1)
        del octree
        return lmk3d_lst


def get_laoin(data_path):
    import mediapipe as mp
    from PIL import Image
    from scipy.spatial.transform import Rotation

    mp_face_mesh = mp.solutions.face_mesh

    image_path = os.path.join(data_path, 'img.png')
    input_image = np.array(Image.open(image_path))

    facemesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5, )

    results = facemesh.process(input_image).multi_face_landmarks

    output_file_path = os.path.join(data_path, 'laionface.txt')

    with open(output_file_path, "w") as file:
        for landmark in results[0].landmark:
            x = landmark.x
            y = landmark.y
            z = landmark.z

            file.write(f"{x} {y} {z}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get RGBA and mask')
    parser.add_argument('-source_dir', type=str, default='/home/haojinkun/3DAvatar/PanoHead-main/dataset/head_img_new')
    parser.add_argument('-mesh_dir', type=str, default='/home/haojinkun/3DAvatar/PanoHead-main/preprocess_data')
    parser.add_argument('-img_name', type=str, default='')
    args = parser.parse_args()

    source_dir = args.source_dir
    mesh_dir = args.mesh_dir


    face_id = args.img_name

    face_name = face_id.split('/')[-1].split('.')[0]

    cam_file = os.path.join(source_dir, 'dataset.json')
    face_file = os.path.join(source_dir, 'face_data.pkl')
    mesh_file = os.path.join(mesh_dir, face_name, 'init_mesh.ply')
    save_path = os.path.join(mesh_dir, face_name)

    with open(cam_file, 'r') as f:
        cam_data = json.load(f)

    with open(face_file, 'rb') as f:
        face_data = pickle.load(f)

    with open(mesh_file, 'rb') as f:
        ply_data = PlyData.read(f)

    face_id_key = os.path.commonprefix(list(face_data.keys())) + face_id

    # get laion face
    data_path = os.path.join(mesh_dir, face_name)
    get_laoin(data_path)
    input_file_path = os.path.join(data_path, 'laionface.txt')
    with open(input_file_path, "r") as file:
        lines = file.readlines()
    coordinates = [list(map(float, line.strip().split())) for line in lines]
    laion_face = np.array(coordinates)

    face_file_name = face_id.split('/')[-1]
    camera = np.array(cam_data['labels'][face_file_name]['label']).astype(np.float64)

    camera_file = os.path.join(save_path, 'camera.npy')
    np.save(camera_file, np.array(cam_data['labels'][face_file_name]['label_fan']).astype(np.float64))

    kps = face_data[face_id_key]['kps']
    if len(kps) == 0:
        print(face_id)
        # continue
    dense_face = face_data[face_id_key]['dense_face']
    sparse_face = face_data[face_id_key]['sparse_face']

    # 7 control points
    source_points = sparse_face[[36, 39, 42, 45, 30, 60, 54], :]
    source_points_laion = laion_face[[33, 133, 362, 263, 4, 62, 29], :]

    cam2world_matrix = camera[:16].reshape((4, 4))
    intrinsics = camera[16:25].reshape((3, 3))

    Lmproj = LdmkProjector()
    target_points = Lmproj.ldmk_3d_detect(ply_data, kps, intrinsics, cam2world_matrix)  # (num_kps, 3)

    # np.savetxt(os.path.join('process_data', 'target_kps.txt'), target_points, fmt='%f %f %f')

    del Lmproj
    # R, S, T = PoseEstimate.svd(source_points, target_points)
    S, T = translation_and_scale(source_points, target_points)

    poseestimate = PoseEstimate()
    R_laion, S_laion, T_laion = poseestimate.ransac_svd(source_points_laion, target_points)

    transform_sparse = sparse_face * S + T
    transform_dense = dense_face * S + T

    transform_laion = np.dot(laion_face * S_laion, R_laion.T) + T_laion

    np.savetxt(os.path.join(save_path, 'laion.txt'), transform_laion, fmt='%f %f %f')
    np.savetxt(os.path.join(save_path, 'sparse.txt'), transform_sparse, fmt='%f %f %f')
    np.savetxt(os.path.join(save_path, 'dense.txt'), transform_dense, fmt='%f %f %f')




