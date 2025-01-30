import cv2
import dlib
import pickle
import numpy as np
from PIL import Image, ImageDraw
import os


def draw_landmarks(image, landmarks, color="white", radius=2.5):
    draw = ImageDraw.Draw(image)
    for dot in landmarks:
        x, y = dot
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)


def get_68landmarks_img(img, image_pil, detector, predictor):
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
    # con_img = Image.new('RGB', (img.shape[1], img.shape[0]), color=(0, 0, 0))
    draw_landmarks(image_pil, landmarks_7)
    image_pil = np.array(image_pil)
    image_pil = cv2.cvtColor(image_pil, cv2.COLOR_BGR2RGB)
    return image_pil, landmarks_7


# load face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# load croped images
img_dir = "crop_samples/img_new"
list_dir = os.listdir(img_dir)

# new dict for keypoints
landmarks_7_dict = {}


if not os.path.exists('landmark_img'):
    os.makedirs('landmark_img',exist_ok=True)

for img_name in list_dir:
    _, extension = os.path.splitext(img_name)

    # only do it for images, not .json file
    if extension == '.json':
        continue
    if extension == '.pkl':
        continue

    img_path = os.path.join(img_dir, img_name)
    image = cv2.imread(img_path)
    image_pil = Image.open(img_path)
    img, landmarks_7 = get_68landmarks_img(image,image_pil,detector,predictor)
    cv2.imwrite(f'landmark_img/{img_name[:-4]}_lm.png', img)

    # gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect face
    rects = detector(gray, 1)

    landmarks_7_dict[img_path] = landmarks_7

# save the data.pkl pickle
with open(os.path.join(img_dir, 'kps_7.pkl'), 'wb') as f:
    pickle.dump(landmarks_7_dict, f)