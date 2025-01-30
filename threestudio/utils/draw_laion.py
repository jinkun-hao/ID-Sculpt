from typing import Mapping
import mediapipe as mp
import numpy as np
from PIL import Image
import cv2
import os
import json
import torch
import threestudio.utils.drawing_utils as drawing_utils
from diffusers.utils import load_image
# import open3d as o3d

class DrawLaion:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        DrawingSpec = mp.solutions.drawing_styles.DrawingSpec

        f_thick = 2
        f_rad = 1
        right_iris_draw = DrawingSpec(color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad)
        right_eye_draw = DrawingSpec(color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad)
        right_eyebrow_draw = DrawingSpec(color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad)
        left_iris_draw = DrawingSpec(color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad)
        left_eye_draw = DrawingSpec(color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad)
        left_eyebrow_draw = DrawingSpec(color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad)
        mouth_draw = DrawingSpec(color=(10, 180, 10), thickness=f_thick, circle_radius=f_rad)
        head_draw = DrawingSpec(color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad)

        # mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
        self.face_connection_spec = {}
        for edge in self.mp_face_mesh.FACEMESH_FACE_OVAL:
            self.face_connection_spec[edge] = head_draw
        for edge in self.mp_face_mesh.FACEMESH_LEFT_EYE:
            self.face_connection_spec[edge] = left_eye_draw
        for edge in self.mp_face_mesh.FACEMESH_LEFT_EYEBROW:
            self.face_connection_spec[edge] = left_eyebrow_draw
        # for edge in mp_face_mesh.FACEMESH_LEFT_IRIS:
        #    face_connection_spec[edge] = left_iris_draw
        for edge in self.mp_face_mesh.FACEMESH_RIGHT_EYE:
            self.face_connection_spec[edge] = right_eye_draw
        for edge in self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
            self.face_connection_spec[edge] = right_eyebrow_draw
        # for edge in mp_face_mesh.FACEMESH_RIGHT_IRIS:
        #    face_connection_spec[edge] = right_iris_draw
        for edge in self.mp_face_mesh.FACEMESH_LIPS:
            self.face_connection_spec[edge] = mouth_draw
        self.iris_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}

        self.face_connection_spec_back = {}
        for edge in self.mp_face_mesh.FACEMESH_FACE_OVAL:
            self.face_connection_spec_back[edge] = head_draw

    def draw_pupils(self, image, landmarks, drawing_spec, halfwidth: int = 2):
        """We have a custom function to draw the pupils because the mp.draw_landmarks method requires a parameter for all
        landmarks.  Until our PR is merged into mediapipe, we need this separate method."""
        if len(image.shape) != 3:
            raise ValueError("Input image must be H,W,C.")
        image_rows, image_cols, image_channels = image.shape
        if image_channels != 3:  # BGR channels
            raise ValueError('Input image must contain three channel bgr data.')
        for idx, (x, y) in enumerate(landmarks):
            image_x = x
            image_y = y
            draw_color = None
            if isinstance(drawing_spec, Mapping):
                if drawing_spec.get(idx) is None:
                    continue
                else:
                    draw_color = drawing_spec[idx].color
            elif isinstance(drawing_spec, DrawingSpec):
                draw_color = drawing_spec.color
            image[image_y-halfwidth:image_y+halfwidth, image_x-halfwidth:image_x+halfwidth, :] = draw_color


    def reverse_channels(self, image):
        """Given a numpy array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
        # im[:,:,::-1] is a neat hack to convert BGR to RGB by reversing the indexing order.
        # im[:,:,::[2,1,0]] would also work but makes a copy of the data.
        return image[:, :, ::-1]


    def draw(self, azimuth, mvp, face_landmarks, H, W):
        image_size = (H, W, 3)
        empty = np.zeros(image_size, dtype=np.uint8)

        landmarks_homogeneous = torch.cat(
            [face_landmarks, torch.ones([face_landmarks.shape[0], 1]).to(face_landmarks)], dim=-1
        )   # torch.Size([478, 4]) torch.Size([1, 4, 4])

        landmarks_projected = torch.matmul(landmarks_homogeneous, mvp.permute(0, 2, 1))

        landmarks_projected /= landmarks_projected[:, :, 3:4]
        landmarks_projected = landmarks_projected[:, :, :2]

        landmarks_projected = landmarks_projected.squeeze().detach().cpu().numpy()
        landmarks_projected = ((landmarks_projected + 1) / 2 * image_size[0]).astype(int)

        if azimuth < -90 or azimuth > 90:
            drawing_utils.draw_landmarks(
                empty,
                landmarks_projected,
                connections=self.face_connection_spec_back.keys(),
                landmark_drawing_spec=None,
                connection_drawing_spec=self.face_connection_spec_back
            )
        else:
            drawing_utils.draw_landmarks(
                empty,
                landmarks_projected,
                connections=self.face_connection_spec.keys(),
                landmark_drawing_spec=None,
                connection_drawing_spec=self.face_connection_spec
            )
            self.draw_pupils(empty, landmarks_projected, self.iris_landmark_spec, 2)

        # Flip BGR back to RGB.
        empty = self.reverse_channels(empty)

        return empty


if __name__ == '__main__':
    face_id = 'id'
    dataset_dir = f'path_to_the_dataset/{face_id}'

    input_file_path = os.path.join(dataset_dir, 'laion.txt')

    with open(input_file_path, "r") as file:
        lines = file.readlines()

    coordinates = [list(map(float, line.strip().split())) for line in lines]
    face_landmarks = np.array(coordinates)
    import pdb

    pdb.set_trace()

    # Project 3D landmark to the image coordinates
    landmarks_homogeneous = np.concatenate([face_landmarks, np.ones((face_landmarks.shape[0], 1))], axis=1)
    landmarks_projected = mvp @ landmarks_homogeneous.T
    landmarks_projected /= landmarks_projected[3, :].clone()
    landmarks_projected = landmarks_projected[:2, :].T

    landmarks_projected = landmarks_projected.cpu().numpy()
    # Map the image coordinates to the image range
    landmarks_projected = ((landmarks_projected + 1) / 2 * image_size[0]).astype(int)

    img_rgb = load_image('test_img.png')

    empty = np.zeros_like(img_rgb)

    # Draw detected faces:
    drawing_utils.draw_landmarks(
        empty,
        landmarks_projected,
        connections=face_connection_spec.keys(),
        landmark_drawing_spec=None,
        connection_drawing_spec=face_connection_spec
    )
    draw_pupils(empty, landmarks_projected, iris_landmark_spec, 2)

    # Flip BGR back to RGB.
    empty = reverse_channels(empty)
    # cv2.imwrite('output_laion.png', empty)





