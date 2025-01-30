# coding: utf-8

import os
import sys
import argparse
import cv2
import yaml
import glob
from tqdm import tqdm
import numpy as np
import json
import dlib
import pickle

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
#from utils.render_ctypes import render  # faster
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import calc_pose, P2sRt, matrix2angle
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool

def eg3dcamparams(R_in):
    camera_dist = 2.7
    intrinsics = np.array([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
    # assume inputs are rotation matrices for world2cam projection
    R = np.array(R_in).astype(np.float32).reshape(4,4)
    # add camera translation
    t = np.eye(4, dtype=np.float32)
    t[2, 3] = - camera_dist

    # convert to OpenCV camera
    convert = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]).astype(np.float32)

    # world2cam -> cam2world
    P = convert @ t @ R
    cam2world = np.linalg.inv(P)

    convert_fan = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]).astype(np.float32)
    # world2cam -> cam2world
    P_fan = convert_fan @ t @ R
    cam2world_fan = np.linalg.inv(P_fan)

    # add intrinsics
    label_new = np.concatenate([cam2world.reshape(16), intrinsics.reshape(9)], -1)
    label_fan = np.concatenate([cam2world_fan.reshape(16), intrinsics.reshape(9)], -1)
    return {'label':label_new, 'label_fan':label_fan}

def get_crop_bound(lm, method="ffhq"):
    if len(lm) == 106:
        left_e = lm[104]
        right_e = lm[105]
        nose = lm[49]
        left_m = lm[84]
        right_m = lm[90]
        center = (lm[1] + lm[31]) * 0.5
    elif len(lm) == 68:
        left_e = np.mean(lm[36:42], axis=0)
        right_e = np.mean(lm[42:48], axis=0)
        nose = lm[33]
        left_m = lm[48]
        right_m = lm[54]
        center = (lm[0] + lm[16]) * 0.5
    else:
        raise ValueError(f"Unknown type of keypoints with a length of {len(lm)}")

    if method == "ffhq":
        eye_to_eye = right_e - left_e
        eye_avg = (left_e + right_e) * 0.5
        mouth_avg = (left_m + right_m) * 0.5
        eye_to_mouth = mouth_avg - eye_avg
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
    elif method == "default":
        eye_to_eye = right_e - left_e
        eye_avg = (left_e + right_e) * 0.5
        eye_to_nose = nose - eye_avg
        x = eye_to_eye.copy()
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.4, np.hypot(*eye_to_nose) * 2.75)
        y = np.flipud(x) * [-1, 1]
        c = center
    else:
        raise ValueError('%s crop method not supported yet.' % method)
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    return quad.astype(np.float32), c, x, y

def crop_image(img, mat, crop_w, crop_h, upsample=1, borderMode=cv2.BORDER_CONSTANT):
    crop_size = (crop_w, crop_h)
    if upsample is None or upsample == 1:
        crop_img = cv2.warpAffine(np.array(img), mat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
    else:
        assert isinstance(upsample, int)
        crop_size_large = (crop_w*upsample,crop_h*upsample)
        crop_img = cv2.warpAffine(np.array(img), upsample*mat, crop_size_large, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
        crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA) 
    return crop_img

def process_face(points):
    # 1. 平移点云使得中心位于原点
    center = np.mean(points, axis=0)
    points_centered = points - center
    # sparse_points_centered = sparse_points - center

    # 2. 计算缩放因子并缩放点云到单位盒子
    scale_factor = 1.0 / np.max(np.abs(points_centered))
    points_normalized = points_centered * scale_factor
    # sparse_points_normalized = sparse_points_centered * scale_factor

    # 3. 旋转点云180度沿 Z 轴
    rotation_matrix = np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
                                [np.sin(np.pi), np.cos(np.pi), 0],
                                [0, 0, 1]])
    points_output = np.dot(points_normalized, rotation_matrix)
    # sparse_points_output = np.dot(sparse_points_normalized, rotation_matrix)

    return points_output

def get_landmarks(img, detector, predictor):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    landmarks_7 = []
    for face in faces:
        shape = predictor(gray, face)
        # for i in range(68):
        for i in [36, 39, 42, 45, 30, 60, 54]:
            x = shape.part(i).x
            y = shape.part(i).y
            landmarks_7.append((x, y))

    # draw 7 kps on img
    # for landmarks in landmarks_7:
    # for (x, y) in landmarks_7:
    #     cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
    # cv2.imwrite('img_w_kps.png', img)

    return np.array(landmarks_7)

def crop_final(
    img,
    size=1024,
    quad=None,
    top_expand=0.1, 
    left_expand=0.05, 
    bottom_expand=0.0, 
    right_expand=0.05, 
    blur_kernel=None,
    borderMode=cv2.BORDER_REFLECT,
    upsample=2,
    min_size=256,
):  

    orig_size = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
    if min_size is not None and orig_size < min_size:
        return None

    crop_w = int(size * (1 + left_expand + right_expand))
    crop_h = int(size * (1 + top_expand + bottom_expand))
    crop_size = (crop_w, crop_h)
    
    top = int(size * top_expand)
    left = int(size * left_expand)
    size -= 1
    bound = np.array([[left, top], [left, top + size], [left + size, top + size], [left + size, top]],
                        dtype=np.float32)

    mat = cv2.getAffineTransform(quad[:3], bound[:3])
    if upsample is None or upsample == 1:
        crop_img = cv2.warpAffine(np.array(img), mat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
    else:
        assert isinstance(upsample, int)
        crop_size_large = (crop_w*upsample,crop_h*upsample)
        crop_img = cv2.warpAffine(np.array(img), upsample*mat, crop_size_large, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
        crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA) 

    empty = np.ones_like(img) * 255
    crop_mask = cv2.warpAffine(empty, mat, crop_size)



    if True:
        mask_kernel = int(size*0.02)*2+1
        blur_kernel = int(size*0.03)*2+1 if blur_kernel is None else blur_kernel
        downsample_size = (crop_w//8, crop_h//8)
        
        if crop_mask.mean() < 255:
            blur_mask = cv2.blur(crop_mask.astype(np.float32).mean(2),(mask_kernel,mask_kernel)) / 255.0
            blur_mask = blur_mask[...,np.newaxis]#.astype(np.float32) / 255.0
            blurred_img = cv2.blur(crop_img, (blur_kernel, blur_kernel), 0)
            crop_img = crop_img * blur_mask + blurred_img * (1 - blur_mask)
            crop_img = crop_img.astype(np.uint8)
    
    return crop_img

def find_center_bbox(roi_box_lst, w, h):
    bboxes = np.array(roi_box_lst)
    dx = 0.5*(bboxes[:,0] + bboxes[:,2]) - 0.5*(w-1)
    dy = 0.5*(bboxes[:,1] + bboxes[:,3]) - 0.5*(h-1)
    dist = np.stack([dx,dy],1)
    return np.argmin(np.linalg.norm(dist, axis=1))


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()


    with open(args.input_path, "rb") as f:
        inputs = pickle.load(f, encoding="latin1").items()

    # load face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    face_data = {}
    size = 1024
    results_quad = {}
    results_meta = {}
    for i,item in enumerate(tqdm(inputs)):

        # get initial cropping box(quad) using landmarks
        img_path, landmarks = item
        face_data[img_path] = {}
        img_path = args.prefix + img_path
        img_orig = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        if img_orig is None:
            print(f'Cannot load image')
            continue
        quad, quad_c, quad_x, quad_y = get_crop_bound(landmarks)    # 获取裁剪边界

        skip = False
        for iteration in range(1):
            bound = np.array([[0, 0], [0, size-1], [size-1, size-1], [size-1, 0]], dtype=np.float32)   # 初始化相机平面边界
            mat = cv2.getAffineTransform(quad[:3], bound[:3])   # 获取仿射变换矩阵
            img = crop_image(img_orig, mat, size, size)
            h, w = img.shape[:2]

            # Detect faces, get 3DMM params and roi boxes
            boxes = face_boxes(img)
            if len(boxes) == 0:
                print(f'No face detected')
                skip = True
                break

            # only detect one face in the image
            param_lst, roi_box_lst = tddfa(img, boxes)
            box_idx = find_center_bbox(roi_box_lst, w, h)
            dense_face = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True, rotate_flag=True)
            # sparse_face = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False, rotate_flag=True)

            dense_face= process_face(dense_face[box_idx].T)
            ldmrk_indx = tddfa.bfm.keypoints.reshape(-1,3)[:,0]/3
            sparse_face = dense_face[ldmrk_indx.astype(int), :]
            face_data[img_path]['dense_face'] = dense_face     # (38365, 3)
            face_data[img_path]['sparse_face'] = sparse_face   # (68, 3)

            param = param_lst[box_idx]
            P = param[:12].reshape(3, -1)  # camera matrix
            s_relative, R, t3d = P2sRt(P)
            pose = matrix2angle(R)
            pose = [p * 180 / np.pi for p in pose]

            # Adjust z-translation in object space
            R_ = param[:12].reshape(3, -1)[:, :3]
            u = tddfa.bfm.u.reshape(3, -1, order='F')
            trans_z = np.array([ 0, 0, 0.5*u[2].mean() ]) # Adjust the object center
            trans = np.matmul(R_, trans_z.reshape(3,1))
            t3d += trans.reshape(3)

            ''' Camera extrinsic estimation for GAN training '''
            # Normalize P to fit in the original image (before 3DDFA cropping)
            sx, sy, ex, ey = roi_box_lst[0]
            scale_x = (ex - sx) / tddfa.size
            scale_y = (ey - sy) / tddfa.size
            t3d[0] = (t3d[0]-1) * scale_x + sx
            t3d[1] = (tddfa.size-t3d[1]) * scale_y + sy
            t3d[0] = (t3d[0] - 0.5*(w-1)) / (0.5*(w-1)) # Normalize to [-1,1]
            t3d[1] = (t3d[1] - 0.5*(h-1)) / (0.5*(h-1)) # Normalize to [-1,1], y is flipped for image space
            t3d[1] *= -1
            t3d[2] = 0 # orthogonal camera is agnostic to Z (the model always outputs 66.67)

            s_relative = s_relative * 2000
            scale_x = (ex - sx) / (w-1)
            scale_y = (ey - sy) / (h-1)
            s = (scale_x + scale_y) / 2 * s_relative
            # print(f"[{iteration}] s={s} t3d={t3d}")

            if s < 0.7 or s > 1.3:
                print(f"Skipping[{i+1-len(results_quad)}/{i+1}]: {img_path} s={s}")
                skip = True
                break
            if abs(pose[0]) > 90 or abs(pose[1]) > 80 or abs(pose[2]) > 50:
                print(f"Skipping[{i+1-len(results_quad)}/{i+1}]: {img_path} pose={pose}")
                skip = True
                break
            if abs(t3d[0]) > 1. or abs(t3d[1]) > 1.:
                print(f"Skipping[{i+1-len(results_quad)}/{i+1}]: {img_path} pose={pose} t3d={t3d}")
                skip = True
                break

            quad_c = quad_c + quad_x * t3d[0]
            quad_c = quad_c - quad_y * t3d[1]
            quad_x = quad_x * s
            quad_y = quad_y * s
            c, x, y = quad_c, quad_x, quad_y
            quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(np.float32)
            

        if skip:
            continue

        # final projection matrix
        s = 1
        t3d = 0 * t3d
        R[:,:3] = R[:,:3] * s
        P = np.concatenate([R,t3d[:,None]],1)
        P = np.concatenate([P, np.array([[0,0,0,1.]])],0)
        results_meta[img_path] = eg3dcamparams(P.flatten())
        results_quad[img_path] = quad

        cropped_img = crop_final(img_orig, size=size, quad=quad)
        face_data[img_path]['kps'] = get_landmarks(cropped_img, detector, predictor)
        os.makedirs(args.out_dir, exist_ok=True)
        print(f'{args.out_dir}   {os.path.basename(img_path).replace(".jpg",".png")}    {cropped_img.shape}')
        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(img_path).replace(".jpg",".png")), cropped_img)

    # Save quads
    print("results:", len(results_quad))
    with open(args.output, 'wb') as f:
        pickle.dump(results_quad, f)

    # Save meta data
    results_new = {}
    for key, value  in results_meta.items():
        img = os.path.basename(key)
        formatted_data = {key: ['{:.6f}'.format(value) for value in array] for key, array in value.items()}
        # res = [format(P[r], '.6f') for r in P.keys()]
        results_new[img] = formatted_data

        # results_new.append((img,res))
    with open(os.path.join(args.out_dir, args.output_json), 'w') as outfile:
        json.dump({"labels": results_new}, outfile, indent="\t")

    with open(os.path.join(args.out_dir, 'face_data.pkl'), 'wb') as f:
        pickle.dump(face_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-i', '--input_path', type=str, default='data.pkl')
    parser.add_argument('-o', '--output', type=str, default='quads.pkl')
    parser.add_argument('-j', '--output_json', type=str, default='dataset.json')
    parser.add_argument('-p', '--prefix', type=str, default='')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--out_dir', type=str, default='./crop_samples/img_new')
    parser.add_argument('--mode', type=str, default='gpu', help='gpu or cpu mode')
    parser.add_argument('--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('--individual', action='store_true', default=False)
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)

    
