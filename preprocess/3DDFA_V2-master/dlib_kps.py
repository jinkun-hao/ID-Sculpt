import cv2
import dlib
import pickle
import numpy as np
import os
import argparse

def main(args):
    # load face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # load images
    img_dir = args.input_dir
    list_dir = os.listdir(img_dir)

    # new dict for keypoints
    landmarks = {}

    for img_name in list_dir:
        _, extension = os.path.splitext(img_name)

        # only do it for images, not .json file
        if extension == '.json':
            continue

        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)

        # gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect face
        rects = detector(gray, 1)



        for (i, rect) in enumerate(rects):
            # get keypoints
            shape = predictor(gray, rect)

            # save kps to the dict
            landmarks[img_path] = [np.array([p.x, p.y]) for p in shape.parts()]

    # save the data.pkl pickle
    with open(args.output_path, 'wb') as f:
        pickle.dump(landmarks, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of kps')
    parser.add_argument('-input_dir', type=str, default="test/head_img_new")
    parser.add_argument('-output_path', type=str, default='data.pkl')
    args = parser.parse_args()
    main(args)
