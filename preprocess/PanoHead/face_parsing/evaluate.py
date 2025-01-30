#!/usr/bin/python
# -*- encoding: utf-8 -*-

# from logger import setup_logger
from model import BiSeNet
from face_dataset import FaceMask

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import argparse
import os.path as osp
# import logging
import time
import numpy as np
from tqdm import tqdm
import math
from PIL import Image
import torchvision.transforms as transforms
import cv2

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        # vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
        vis_parsing_anno_color[index[0], index[1], :] = [255, 255, 255]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def save_mask(im, parsing_anno, stride, save_path='path/to/save/mask.png', face_mask_path='path/to/save/facemask.png'):
    # Create a binary mask with white for foreground and black for background
    mask = np.zeros_like(parsing_anno, dtype=np.uint8)
    mask[parsing_anno > 0] = 255

    # Resize the mask to the original image size
    mask = cv2.resize(mask, (im.width, im.height), interpolation=cv2.INTER_NEAREST)

    kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed
    mask = cv2.erode(mask, kernel, iterations=1)

    face_mask = mask * (parsing_anno != 17) * (parsing_anno != 18) * (parsing_anno != 16) * (parsing_anno != 15) * (parsing_anno != 14) * (parsing_anno != 7) * (parsing_anno != 8) * (parsing_anno != 6)
    # Save the mask
    cv2.imwrite(save_path, mask)
    cv2.imwrite(face_mask_path, face_mask)
    return mask

def evaluate(args, respth='./res/test_res', dspth='./data', cp='79999_iter.pth'):
    respth = args.output_dir
    dspth = args.input_dir

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = cp
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in tqdm(os.listdir(dspth)):
            _, extension = os.path.splitext(image_path)
            # import pdb;pdb.set_trace()
            # only do it for images, not .json file
            if extension == '.json':
                continue
            if extension == '.pkl':
                continue

            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            if not osp.exists(osp.join(respth, os.path.splitext(image_path)[0])):
                os.makedirs(osp.join(respth, os.path.splitext(image_path)[0]))
            
            original_image_path = osp.join(respth, f'{os.path.splitext(image_path)[0]}', 'img.png')
            image.save(original_image_path)

            mask_path = osp.join(respth, f'{os.path.splitext(image_path)[0]}', 'mask.png')
            # vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))
            face_mask_path = osp.join(respth, f'{os.path.splitext(image_path)[0]}', 'face_mask.png')
            mask = save_mask(image, parsing, stride=1, save_path=mask_path, face_mask_path=face_mask_path)

            # masked_image_path = osp.join(respth, os.path.splitext(image_path)[0], 'img_masked.png')


            vis_im = cv2.imread(original_image_path)

            masked_image = vis_im * (mask[:, :, np.newaxis] / 255) + (255 - mask[:, :, np.newaxis])

            # cv2.imwrite(masked_image_path, masked_image)
            # cv2.imwrite(masked_crop_path, masked_image)

            mask_rgb = np.repeat(mask[:,:,np.newaxis], 1, axis=2)
            rgba_img = np.concatenate([vis_im, mask_rgb], axis=2)[:,:,[2,1,0,3]]
            rgba_pil = Image.fromarray(rgba_img)
            rgba_image_path = osp.join(respth, f'{os.path.splitext(image_path)[0]}', 'img_rgba.png')
            rgba_pil.save(rgba_image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get RGBA and mask')
    parser.add_argument('-input_dir', type=str, default='/home/haojinkun/3DAvatar/3DDFA_V2-master/test/head_img_new')
    parser.add_argument('-output_dir', type=str, default='/home/haojinkun/3DAvatar/PanoHead-main/preprocess_data')
    args = parser.parse_args()
    evaluate(args)
