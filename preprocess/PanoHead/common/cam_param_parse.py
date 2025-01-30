import numpy as np 
import yaml 
import xml.dom.minidom
from PIL import Image, ExifTags

from common.transform.transform import quat_rotation_mat

def parse_cam_param(xml_path, cam_idx_lst):
	root = xml.dom.minidom.parse(xml_path)
	cam_params = root.getElementsByTagName('camera')
	sensor_params = root.getElementsByTagName('sensor')

	cam_mat_lst = []
	trans_mat_lst = []
	valid_lst = []

	for cam_idx in cam_idx_lst:
		if '.jpg' in cam_idx or '.JPG' in cam_idx or '.png' in cam_idx or '.PNG' in cam_idx:
			cam_idx = cam_idx[:-4]
		cam_param = None
		sensor_param = None

		for p in cam_params:
			if p.getAttribute('label') == cam_idx:
				cam_param = p
				break

		sensor_id = cam_param.getAttribute('sensor_id')
		for p in sensor_params:
			if p.getAttribute('id') == sensor_id:
				sensor_param = p 
				break

		width = int(sensor_param.getElementsByTagName('resolution')[0].getAttribute('width'))
		height = int(sensor_param.getElementsByTagName('resolution')[0].getAttribute('height'))
	
		f = float(sensor_param.getElementsByTagName('f')[0].firstChild.data)
		cx = float(sensor_param.getElementsByTagName('cx')[0].firstChild.data)
		cy = float(sensor_param.getElementsByTagName('cy')[0].firstChild.data)

		cam_mat = np.zeros((3, 3))
		cam_mat[2, 2] = 1.0

		if width > height:
			cam_mat[0, 1] = -f 
			cam_mat[1, 0] = f 
			cam_mat[0, 2] = height / 2 - cy
			cam_mat[1, 2] = width / 2 + cx
		else:
			cam_mat[0, 0] = f 
			cam_mat[1, 1] = f 
			cam_mat[0, 2] = width / 2 + cx 
			cam_mat[1, 2] = height / 2 + cy

		if len(cam_param.getElementsByTagName('transform')) == 0:
			valid_lst.append(False)
			continue

		transform = cam_param.getElementsByTagName('transform')[0].firstChild.data
		ss = transform.split(' ')
		trans_mat = np.zeros((4, 4))
		for i in range(4):
			for j in range(4):
				trans_mat[i, j] = float(ss[i * 4 + j])

		trans_mat = np.linalg.inv(trans_mat)

		cam_mat_lst.append(cam_mat) 
		trans_mat_lst.append(trans_mat)
		valid_lst.append(True)
	
	return np.array(cam_mat_lst), np.array(trans_mat_lst), np.array(valid_lst)

def get_exif_mat(img_name, w, h):
	exif_mat = np.identity(3)
	try:
		img_pil = Image.open(img_name)
	except:
		print('read fail ', img_name)
		return exif_mat
	if img_pil is None:
		print('img_pil none ', img_name)
		return exif_mat
	for orientation in ExifTags.TAGS.keys(): 
		if ExifTags.TAGS[orientation] == 'Orientation': 
			break 
	if img_pil._getexif() is None:
		return exif_mat
	
	exif = dict(img_pil._getexif().items())
	if not exif.__contains__(orientation):
		return exif_mat
	exif_mat = np.zeros((3, 3), dtype = np.float32)
	exif_mat[2, 2] = 1 

	if exif[orientation] == 1 :  #IMAGE_ORIENTATION_TL = 1, ///< Horizontal (normal)
		exif_mat[0, 0] = 1 
		exif_mat[1, 1] = 1 
	elif exif[orientation] == 2 :  #Mirrored horizontal [u,v]->w-u,v] IMAGE_ORIENTATION_TR = 2, ///< Mirrored horizontal
		exif_mat[0, 0] = -1  
		exif_mat[0, 2] = w 
		exif_mat[1, 1] = 1
	elif exif[orientation] == 3 :  #180 [u,v]->[w-u,h-v] IMAGE_ORIENTATION_BR = 3, ///< Rotate 180
		exif_mat[0, 0] = -1 
		exif_mat[0, 2] = w 
		exif_mat[1, 1] = -1
		exif_mat[1, 2] = h
	elif exif[orientation] == 4 :  #Mirrored vertical [u,v]->[u,h-v] IMAGE_ORIENTATION_BL = 4, ///< Mirrored vertical
		exif_mat[0, 0] = 1 
		exif_mat[1, 1] = -1 
		exif_mat[1, 2] = h
	elif exif[orientation] == 5 :  # ?? maybe error Mirrored horizontal & rotate 270 CW [u,v]->[u,h-v] IMAGE_ORIENTATION_LT = 5, ///< Mirrored horizontal & rotate 270 CW
		exif_mat[0, 0] = 1 
		exif_mat[0, 1] = -1 
		exif_mat[1, 2] = w
	elif exif[orientation] == 6 :  #90 [u,v]->[h-v,u] IMAGE_ORIENTATION_RT = 6, ///< Rotate 90 CW
		exif_mat[0, 1] = -1
		exif_mat[1, 0] = 1 
		exif_mat[0, 2] = h
	elif exif[orientation] == 7 :  #meybe error Mirrored horizontal & rotate 90 CW IMAGE_ORIENTATION_RB = 7, ///< Mirrored horizontal & rotate 90 CW
		exif_mat[0, 1] = 1 
		exif_mat[1, 0] = 1
		exif_mat[0, 2] = w
		exif_mat[1, 2] = h
	elif exif[orientation] == 8 :  #270 [u,v]->[v, w-u] IMAGE_ORIENTATION_LB = 8  ///< Rotate 270 CW
		exif_mat[0, 1] = 1 
		exif_mat[1, 0] = -1 
		exif_mat[1, 2] = w 

	return exif_mat

def parse_cam_param_yaml(param_path, img_names, img_root):
	file = open(param_path, 'r')
	data = yaml.safe_load(file)
	num = int(data['num_calibs'])

	cam_mat_lst, trans_mat_lst, size_lst = [], [], []
	valid_lst = []

	for i in range(len(img_names)):
		cnt = -1
		for j in range(num):
			if data['calib_' + str(j)]['image_name'] == img_names[i]:
				cnt = j 
				break
		if cnt == -1:
			valid_lst.append(False)
			continue

		qvec = np.array(data['calib_' + str(cnt)]['image_qvec'])
		r = quat_rotation_mat(qvec)
		t = data['calib_' + str(cnt)]['image_tvec']
		trans_mat = np.eye(4, dtype = np.float32)
		trans_mat[:3, :3] = r 
		trans_mat[:3, 3] = t 
		trans_mat_lst.append(trans_mat)

		size = data['calib_' + str(cnt)]['camera_size']

		cam_param = data['calib_' + str(cnt)]['camera_params'].split(', ')
		cam_mat = np.eye(3, dtype = np.float32)

		cam_mat[0, 0] = float(cam_param[0])
		cam_mat[1, 1] = float(cam_param[0])
		cam_mat[0, 2] = float(cam_param[1])
		cam_mat[1, 2] = float(cam_param[2])

		exif_mat = get_exif_mat(img_root + img_names[i], size[0], size[1])
		cam_mat = exif_mat.dot(cam_mat)

		cam_mat_lst.append(cam_mat)
		valid_lst.append(True)

	return np.array(cam_mat_lst), np.array(trans_mat_lst), np.array(valid_lst)