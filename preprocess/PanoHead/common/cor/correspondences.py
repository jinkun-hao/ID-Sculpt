import numpy as np 
from sklearn.neighbors import NearestNeighbors

import pycuda.driver as cuda 
from pycuda.autoinit import context

import trimesh 
from trimesh.triangles import points_to_barycentric

from common.mesh import TriMesh, TriMesh_cuda 
from common.cor.intersect import Intersect, Intersect_cuda
from common.cor.octree import OCTree
from common.utils import transfer_data_to_gpu

class Correspondences:
	def __init__(self):
		self.tester = Intersect()
		self.tester_cuda = Intersect_cuda()

	def nearest_point(self, src_verts, tgt_verts, neigh = None, threshold = 0.03):
		if neigh is None:
			neigh = NearestNeighbors(n_neighbors = 1)
			neigh.fit(tgt_verts)
		distances, indices = neigh.kneighbors(src_verts, return_distance = True)
		valid_ind = np.where(distances < threshold)[0]
		return valid_ind, indices.flatten()[valid_ind]

	def points_segs_project(self, points, segs_start, segs_end):
		vbva = segs_start - segs_end
		vbvp = points - segs_end
		u = np.sum(vbvp * vbva, axis = 1) / np.sum(vbva * vbva, axis = 1)
		perpendicular = u[:, np.newaxis] * segs_start + (1 - u)[:, np.newaxis] * segs_end
		dist = np.sqrt(np.sum((points - perpendicular) ** 2, axis = 1))
		return u, dist 

	def nearest_tri(self, src_verts, tgt_mesh, dist_threshold = 0.03):
		if not isinstance(tgt_mesh, trimesh.base.Trimesh):
			tgt = trimesh.Trimesh(vertices = tgt_mesh.vertices, faces = tgt_mesh.faces, process = False)
		else:
			tgt = tgt_mesh

		closest_points, distances, face_indices = tgt.nearest.on_surface(src_verts)
		valid_ind = np.where(distances < dist_threshold)[0]
		weights = points_to_barycentric(tgt_mesh.vertices[tgt_mesh.faces[face_indices]], src_verts)
		return valid_ind, face_indices[valid_ind], weights[valid_ind]

	def nearest_tri_normal(self, src, tgt, octree = None, con_ind = None, dist_threshold = 0.03, normal_threshold = 0.9):
		if src.vert_normal is None:
			src.cal_vert_normal()
		tgt.cal_face_normal()

		if octree is None:
			octree = OCTree()
			octree.from_triangles(tgt.vertices, tgt.faces, np.arange(tgt.face_num()))

		ray_ind, face_ind, weights, dist = self.tester.rays_octree_intersect(octree, \
			tgt.vertices, tgt.faces, tgt.face_normal, \
			src.vertices, src.vert_normal, np.arange(src.vert_num()), dist_threshold, normal_threshold)
		weights = np.reshape(weights, (-1, 3))

		tgt_face_ind = np.full(src.vert_num(), -1, dtype = np.int)
		tgt_weights = np.full((src.vert_num(), 3), 0, dtype = np.float32)
		tgt_dist = np.full(src.vert_num(), np.inf, dtype = np.float32)

		for i in range(len(ray_ind)):
			if not con_ind is None: 
				if ray_ind[i] in con_ind and face_ind[i] in con_ind[ray_ind[i]]:
					continue
					
			if tgt_dist[ray_ind[i]] > dist[i]:
				tgt_dist[ray_ind[i]] = dist[i]
				tgt_face_ind[ray_ind[i]] = face_ind[i]
				tgt_weights[ray_ind[i]] = weights[i]

		src_ind = np.where(tgt_dist < dist_threshold)[0]

		return src_ind, tgt_face_ind[src_ind], tgt_weights[src_ind]

	def nearest_tri_normal_gpu(self, src_cuda, tgt_cuda, octree_cuda, dist_threshold = 0.03, normal_threshold = 0):
		if not type(dist_threshold) is np.ndarray:
			dist_thres_lst = np.ones(src_cuda.vert_num, dtype = np.float32) * dist_threshold
		else:
			dist_thres_lst = dist_threshold

		if not type(normal_threshold) is np.ndarray:
			norm_thres_lst = np.ones(src_cuda.vert_num, dtype = np.float32) * normal_threshold
		else:
			norm_thres_lst = normal_threshold

		int_face_ind, int_weights = self.tester_cuda.rays_octree_intersect_cuda(octree_cuda, src_cuda, tgt_cuda, dist_thres_lst, norm_thres_lst)
		
		valid_ind = np.where(int_face_ind >= 0)[0]
		return valid_ind, int_face_ind[valid_ind], int_weights[valid_ind]