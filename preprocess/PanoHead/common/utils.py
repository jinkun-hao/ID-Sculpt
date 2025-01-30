import numpy as np 
import cv2 as cv
import pycuda.driver as cuda 
import xml.dom.minidom

def transfer_data_to_gpu(data):
	data_gpu = cuda.mem_alloc(data.nbytes)
	cuda.memcpy_htod(data_gpu, data)
	return data_gpu

def PCA(pc):
	center = pc - np.mean(pc, axis = 0)
	U, s, V = np.linalg.svd(center)
	W = V.T 
	return W

def crop_img(img, lmk, width_pratio = 0.25, height_pratio = 1):
	h, w = img.shape[:2]
	top = np.min(lmk[:, 1])
	bottom = np.max(lmk[:, 1])
	left = np.min(lmk[:, 0])
	right = np.max(lmk[:, 0])

	left = int(max(0.0, left - (right - left) * width_pratio))
	right = int(min(w, right + (right - left) * width_pratio))
	top = int(max(0.0, top - (bottom - top) * height_pratio))
	bottom = int(min(h, bottom + (bottom - top) * height_pratio))

	return img[top: bottom, left: right]

def scale_img(img, new_h):
	scale = img.shape[0] / new_h
	new_w = int(img.shape[1] * new_h / img.shape[0])
	img = cv.resize(img, (new_w, new_h), cv.INTER_CUBIC)
	return img, scale

def concat_imgs(imgs, num):
    h,w=imgs[0].shape[:2]

    new_h=h*int(np.ceil(len(imgs)/num))
    new_w=num*w
    if len(imgs)<num:
        new_w=w*len(imgs)

    img=np.zeros((new_h,new_w,3))
    cnt=0
    for i in range(len(imgs)):
        row=cnt//num
        col=cnt%num
        if imgs[i].shape[0]==0:
            continue
        img[row*h:(row+1)*h,col*w:(col+1)*w,:] = imgs[i]
        cnt+=1

    return img 

def map_num_to_colors(num, vmin = None, vmax = None):
	import matplotlib.colors as colors
	import matplotlib.cm as cm 

	norm_min = np.min(num) if vmin is None else vmin
	norm_max = np.max(num) if vmax is None else vmax
	norm = colors.Normalize(vmin = norm_min, vmax = norm_max)

	jet = cm.get_cmap('jet')
	scalar = cm.ScalarMappable(norm = norm, cmap = jet)

	return scalar.to_rgba(num)[:, :3]

def plt_init():
	import matplotlib.pyplot as plt
	plt.clf()
	ax=plt.gca()
	ax.xaxis.set_ticks_position('top')
	# ax.invert_yaxis()
	plt.axis('equal')

class KMeans:
	def __init__(self):
		pass

	def initialize_centroids(self, points, K):
		centroids = points.copy()
		np.random.shuffle(centroids)
		return centroids[:K]

	def closest_centroids(self, points, centroids):
		distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
		return np.argmin(distances, axis = 0)

	def move_centroids(self, points, closest, centroids):
		new_centroids = []
		for k in range(centroids.shape[0]):
			new_centroids.append(points[closest == k].mean(axis = 0))

		return np.array(new_centroids)

	def kmeans(self, points, K = 200, max_iters = 10):
		centroids = self.initialize_centroids(points, K)

		for i in range(max_iters):
			closest = self.closest_centroids(points, centroids)
			new_centroids = self.move_centroids(points, closest, centroids)
			if (new_centroids == centroids).all():
				break
			centroids = new_centroids
		return centroids
