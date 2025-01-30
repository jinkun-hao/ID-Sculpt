import numpy as np 

# max_pt_num = 2000
# default_grid_num = 2
# max_layer_num = 3

max_pt_num = 500
default_grid_num = 2
max_layer_num = 6

class OCTree:
	def __init__(self, layer = 0):
		self.item_ind = None
		self.childs = []
		self.layer = layer

	def del_unused_grid(self, item_grid_ind, grid_num):
		grid_valid_ind = np.where(np.isin(np.arange(grid_num), item_grid_ind))[0]
		indexer = np.full(grid_num, -1, dtype = np.int32)
		indexer[grid_valid_ind] = np.arange(len(grid_valid_ind))
		item_grid_ind = indexer[item_grid_ind]

		return item_grid_ind, grid_valid_ind

	def from_vertices(self, vertices, vert_ind):
		self.start = (np.min(vertices[vert_ind], axis = 0) - 1e-6).astype(np.float32)
		self.end = (np.max(vertices[vert_ind], axis = 0) + 1e-6).astype(np.float32)
		grid_size = (self.end - self.start) / default_grid_num
		grid_num = default_grid_num ** 3

		if len(vert_ind) <= max_pt_num or self.layer >= max_layer_num:
			self.item_ind = vert_ind
		else:
			verts_grid_ind_xyz = np.floor((vertices[vert_ind] - self.start) / grid_size).astype(np.int32)
			verts_grid_ind = verts_grid_ind_xyz[:, 0] * (default_grid_num ** 2) + verts_grid_ind_xyz[:, 1] * default_grid_num + verts_grid_ind_xyz[:, 2]
			verts_grid_ind, grid_valid_ind = self.del_unused_grid(verts_grid_ind, grid_num)

			for i in range(len(grid_valid_ind)):
				child_octree = OCTree(self.layer + 1)
				child_octree.from_vertices(vertices, vert_ind[verts_grid_ind == i])
				self.childs.append(child_octree)

	def from_triangles(self, vertices, faces, face_ind):
		self.start = (np.min(vertices[faces[face_ind]].min(axis = 1), axis = 0) - 1e-8).astype(np.float32)
		self.end = (np.max(vertices[faces[face_ind]].max(axis = 1), axis = 0) + 1e-8).astype(np.float32) 

		grid_size = (self.end - self.start) / default_grid_num
		grid_num = default_grid_num ** 3

		if len(face_ind) <= max_pt_num or self.layer >= max_layer_num:
			self.item_ind = face_ind
		else:
			faces_grid_ind_xyz = np.floor((vertices[faces[face_ind]].mean(axis = 1) - self.start) / grid_size).astype(np.int32)
			faces_grid_ind = faces_grid_ind_xyz[:, 0] * (default_grid_num ** 2) + faces_grid_ind_xyz[:, 1] * default_grid_num + faces_grid_ind_xyz[:, 2]
			faces_grid_ind, grid_valid_ind = self.del_unused_grid(faces_grid_ind, grid_num)

			for i in range(len(grid_valid_ind)):
				child_octree = OCTree(self.layer + 1)
				child_octree.from_triangles(vertices, faces, face_ind[faces_grid_ind == i])
				self.childs.append(child_octree)

import struct
import pycuda.driver as cuda 

class OCTree_cuda:
	def __init__(self):
		self.childs = []
		self.mem_size = 4 + 8 * 8 + 6 * 4 + 8 + 8 + 4

	def init_octree(self, octree):
		ptr_lst = np.zeros(8, dtype = np.uintp)
		for i in range(len(octree.childs)):
			child = OCTree_cuda()
			ptr_lst[i] = child.init_octree(octree.childs[i])
			self.childs.append(child)

		ind_gpu_ptr = np.uintp(0)
		num = 0
		if not octree.item_ind is None:
			self.ind_gpu = cuda.to_device(octree.item_ind.astype(np.int32))
			ind_gpu_ptr = np.uintp(self.ind_gpu)
			num = len(octree.item_ind)

		packed_args = struct.pack("8Pi3f3fPi", \
			*ptr_lst, np.int32(len(octree.childs)), \
			*(octree.start.astype(np.float32)), *(octree.end.astype(np.float32)), \
			ind_gpu_ptr, np.int32(num)
			)

		self.ptr = cuda.mem_alloc(self.mem_size)
		cuda.memcpy_htod(self.ptr, packed_args)

		return np.uintp(self.ptr)
