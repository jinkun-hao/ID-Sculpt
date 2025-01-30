import numpy as np 
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp

class TriMesh:
	def __init__(self, vertices = None, faces = None):
		self.vertices = vertices
		self.faces = faces
		self.vert_color = None
		self.vert_normal = None

		self.tex_coords = None
		self.faces_tc = None

	def copy(self):
		tmp = TriMesh()
		if not self.vertices is None:
			tmp.vertices = self.vertices.copy()

		if not self.faces is None:
			tmp.faces = self.faces.copy()

		if not self.vert_color is None:
			tmp.vert_color = self.vert_color.copy()

		if not self.vert_normal is None:
			tmp.vert_normal = self.vert_normal.copy()

		if not self.faces_tc is None:
			tmp.faces_tc = self.faces_tc.copy()

		if not self.tex_coords is None:
			tmp.tex_coords = self.tex_coords.copy()

		return tmp

	def load_npz(self, path):
		data = np.load(path, allow_pickle = True)
		self.vertices = data['vertices']
		self.faces = data['faces']
		# self.tex_coords = data['tex_coords']
		# self.faces_tc = data['faces_tc']

	def load(self, path):
		vertices, vert_color, tex_coords, normals = [], [], [], []
		faces, faces_tc = [], []

		lines = open(path, 'r+').readlines()
		for line in lines:
			line = line.strip('\n')
			line = str.replace(line, '  ', ' ')
			ss = line.split(' ')
			if ss[0] == 'v':
				# for i in range(1, 4):
				for i in range(1, 4):
					vertices.append(float(ss[i]))
				if len(ss) == 7:
					for i in range(4, 7):
						vert_color.append(float(ss[i]))
			if ss[0] == 'vt':
				for i in range(1, 3):
					tex_coords.append(float(ss[i]))

			# if ss[0] == 'vn':
				# for i in range(1, 4):
					# normals.append(float(ss[i]))

			if ss[0] == 'f':
				for i in range(1, 4):
					sv = ss[i].split('/')
					faces.append(int(sv[0]) - 1)
					if len(sv) > 1 and not sv[1] == '':
						faces_tc.append(int(sv[1]) - 1)

		self.vertices = np.reshape(vertices, (-1, 3)).astype(np.float32)
		self.faces = np.reshape(faces, (-1, 3)).astype(np.int32)
		# self.vert_normal = np.reshape(normals, (-1, 3))

		if len(tex_coords) > 0: 
			self.tex_coords = np.reshape(tex_coords, (-1, 2)).astype(np.float32)
		if len(vert_color) == len(vertices):
			self.vert_color = np.reshape(vert_color, (-1, 3)).astype(np.float32)
			
		if len(faces_tc) == len(faces):
			self.faces_tc = np.reshape(faces_tc, (-1, 3)).astype(np.int32)

		if (self.faces_tc == self.faces).all():
			self.faces_tc = None

	def save(self, path, mtllib_path = None, with_color = True, with_vt = True, with_normal = False):
		out_file = open(path, 'w+')
		if not mtllib_path is None:
			out_file.write('mtllib ' + mtllib_path +'\n')

		for i in range(len(self.vertices)):
			v = self.vertices[i]
			line = 'v ' + "{:.6f}".format(v[0]) + ' ' + '{:.6f}'.format(v[1]) + ' ' + '{:.6f}'.format(v[2]) 

			if with_color and not self.vert_color is None:
				c = self.vert_color[i]
				line += ' ' + '{:.6f}'.format(c[0]) + ' ' +'{:.6f}'.format(c[1]) + ' ' + '{:.6f}'.format(c[2]) 

			out_file.write(line + '\n')

		if with_normal and not self.vert_normal is None:
			for i in range(len(self.vert_normal)):
				v = self.vert_normal[i]
				line = 'vn ' + "{:.6f}".format(v[0]) + ' ' + '{:.6f}'.format(v[1]) + ' ' + '{:.6f}'.format(v[2]) 
				out_file.write(line + '\n')

		if with_vt and not self.tex_coords is None:
			for t in self.tex_coords:
				line = 'vt ' + '{:.6f}'.format(t[0]) + ' ' + '{:.6f}'.format(t[1])
				out_file.write(line + '\n')

		for i in range(len(self.faces)):
			line = 'f'
			for j in range(3):
				line += ' ' + str(self.faces[i, j] + 1) 
				if not self.faces_tc is None and with_vt:
					line += '/' + str(self.faces_tc[i, j] + 1)
			out_file.write(line + '\n')

	def cal_vert_normal(self):
		self.vert_normal = np.zeros(self.vertices.shape)
		n0 = np.cross(self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]],\
			self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 0]], axis = 1)

		for i in range(len(self.faces)):
			self.vert_normal[self.faces[i]] += n0[i] 
		self.vert_normal /= np.sqrt(np.sum(self.vert_normal ** 2, axis = 1))[:, np.newaxis] + 1e-12

		self.vert_normal = self.vert_normal.astype(np.float32)

	def cal_face_normal(self):
		verts_0 = self.vertices[self.faces[:, 0]]
		verts_1 = self.vertices[self.faces[:, 1]]
		verts_2 = self.vertices[self.faces[:, 2]]

		self.face_normal = np.cross(verts_1 - verts_0, verts_2 - verts_0)
		self.face_normal = self.face_normal / (np.sqrt(np.sum(self.face_normal ** 2, axis = 1))[:, np.newaxis] + 1e-12)

	def vert_num(self):
		return len(self.vertices)

	def face_num(self):
		return len(self.faces)

	def del_by_vert_ind(self, vert_ind):
		indexer = np.full(self.vert_num(), -1)
		indexer[vert_ind] = np.arange(0, vert_ind.shape[0])

		self.vertices = self.vertices[vert_ind]
		face_mask = np.logical_and(np.logical_and(np.in1d(self.faces[:, 0], vert_ind), np.in1d(self.faces[:, 1], vert_ind)), np.in1d(self.faces[:, 2], vert_ind))
		new_faces = self.faces[face_mask]
		self.faces = indexer[new_faces]

		if not self.faces_tc is None:
			self.faces_tc = self.faces_tc[face_mask]
		if not self.tex_coords is None and self.faces_tc is None:
			self.tex_coords = self.tex_coords[vert_ind]
		if not self.tex_coords is None and not self.faces_tc is None:
			ind = np.unique(self.faces_tc.flatten())
			indexer = np.full(len(self.tex_coords), -1)
			indexer[ind] = np.arange(0, len(ind))
			self.tex_coords = self.tex_coords[ind]
			self.faces_tc = indexer[self.faces_tc]

		if not self.vert_color is None:
			self.vert_color = self.vert_color[vert_ind]
		if not self.vert_normal is None:
			self.vert_normal = self.vert_normal[vert_ind]
		if hasattr(self, 'face_normal') and not self.face_normal is None:
			self.face_normal = self.face_normal[face_mask]
	
	def del_by_vert_mask(self, vert_mask):
		self.vertices = self.vertices[vert_mask]
		
		if not self.vert_color is None:
			self.vert_color = self.vert_color[vert_mask]

		if not self.vert_normal is None:
			self.vert_normal = self.vert_normal[vert_mask]
		
		vert_indices = np.where(vert_mask == True)[0]
		face_mask = np.logical_and(np.logical_and(np.in1d(self.faces[:, 0], vert_indices), np.in1d(self.faces[:, 1], vert_indices)), np.in1d(self.faces[:, 2], vert_indices))
		new_faces = self.faces[face_mask]
		
		indexer = np.full(vert_mask.shape[0], -1)
		indexer[vert_indices] = np.arange(0, vert_indices.shape[0])
		self.faces = indexer[new_faces]

		if not self.tex_coords is None and self.faces_tc is None:
			self.tex_coords = self.tex_coords[vert_mask]

		if not self.tex_coords is None and not self.faces_tc is None:
			self.faces_tc = self.faces_tc[face_mask]
			ind = np.unique(self.faces_tc.flatten())
			indexer = np.full(len(self.tex_coords), -1)
			indexer[ind] = np.arange(0, len(ind))
			self.tex_coords = self.tex_coords[ind]
			self.faces_tc = indexer[self.faces_tc]

	def find_edge_verts(self):
		fnum = self.faces.shape[0]
		vnum = self.vertices.shape[0]

		edge_0 = np.sort(np.vstack((self.faces[:, 0], self.faces[:, 1])), axis = 0)
		edge_1 = np.sort(np.vstack((self.faces[:, 1], self.faces[:, 2])), axis = 0)
		edge_2 = np.sort(np.vstack((self.faces[:, 2], self.faces[:, 0])), axis = 0)

		row = np.hstack((edge_0[0], edge_1[0], edge_2[0]))
		col = np.hstack((edge_0[1], edge_1[1], edge_2[1]))

		edge_share_num = sp.coo_matrix((np.ones(fnum * 3, dtype = np.uint8), (row, col)), shape = (vnum, vnum))
		edge_share_num.sum_duplicates()

		single_ind = np.where(edge_share_num.data < 2)[0]
		return edge_share_num.row[single_ind], edge_share_num.col[single_ind]

	def find_sur_verts(self, ind, max_iters = 5):
		connect = np.zeros((self.vert_num(), self.vert_num()), dtype = np.uint8)
		for i in range(3):
			connect[self.faces[:, i], self.faces[:, (i + 1) % 3]] = 1 
			connect[self.faces[:, i], self.faces[:, (i + 2) % 3]] = 1
		sur_ind = ind.copy()
		for i in range(max_iters):
			for j in range(len(sur_ind)):
				sur_ind = np.append(sur_ind, np.where(connect[sur_ind[j]] == 1)[0])
			sur_ind = np.unique(sur_ind)

		return np.array(list(set(sur_ind) - set(ind)))
	
	def subdiv(self, iters = 1, disp = None):
		edges = np.column_stack([self.faces[:, 0], self.faces[:, 1],\
			self.faces[:, 1], self.faces[:, 2], \
			self.faces[:, 2], self.faces[:, 0]\
			]).reshape((-1, 2))
		edges = np.sort(edges, axis = 1)
		_, unique, inverse = np.unique(edges, axis = 0, return_inverse = True, return_index = True)

		mid = self.vertices[edges[unique]].mean(axis = 1)
		mid_idx = inverse.reshape((-1, 3)) + self.vert_num()

		self.faces = np.column_stack([self.faces[:, 0], mid_idx[:, 0], mid_idx[:, 2], \
			mid_idx[:, 0], self.faces[:, 1], mid_idx[:, 1], \
			mid_idx[:, 2], mid_idx[:, 1], self.faces[:, 2], \
			mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2]\
			]).reshape((-1, 3))
		
		self.vertices = np.vstack((self.vertices, mid))

		# if not self.vt is None:
		# 	vt_mid = self.vt[edges[unique]].mean(axis = 1)
		# 	self.vt = np.vstack((self.vt, vt_mid))
		
		if not disp is None:
			disp_mid = disp[edges[unique]].mean(axis = 1)
			disp = np.vstack((disp, disp_mid))
			return disp


import pycuda.driver as cuda 
from pycuda.autoinit import context
from pycuda.compiler import SourceModule

from common.utils import transfer_data_to_gpu

class TriMesh_cuda:
	def __init__(self):
		pass

	def init_cuda_functions(self):
		mod = SourceModule("""
			__device__ void cross(
				const float* v1, const float* v2,
				float* v
			){
				v[0] = v1[1] * v2[2] - v2[1] * v1[2];
				v[1] = v1[2] * v2[0] - v2[2] * v1[0];
				v[2] = v1[0] * v2[1] - v2[0] * v1[1];
			}

			__device__ void normalize(float* vec, int dim){
				float divide = 0;
				for(int i = 0; i < dim; ++i)
					divide += vec[i] * vec[i];
				divide = sqrt(divide);

				for(int i = 0; i < dim; ++i)
					vec[i] = vec[i] / divide;
			}

			__global__ void cal_face_normal(
				const float* vertices, const int* faces, 
				float* face_normal, int vnum, int fnum
			){
				for(int i = threadIdx.x; i < fnum; i += blockDim.x){
					const float* v0 = vertices + faces[i * 3] * 3;
					const float* v1 = vertices + faces[i * 3 + 1] * 3;
					const float* v2 = vertices + faces[i * 3 + 2] * 3;

					float v0v1[3]{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
					float v0v2[3]{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};

					cross(v0v1, v0v2, face_normal + i * 3);
					normalize(face_normal + i * 3, 3);
				}
			}

			__global__ void norm_vert_normal(float* vn, int vnum){
 				for(int i = threadIdx.x; i < vnum; i += blockDim.x){
 					normalize(vn + i * 3, 3);
 				}
 			}

 			__global__ void cal_vert_normal(
 				const float* vertices, const int* faces, 
 				float* vert_normal, int vnum, int fnum
 			){
 				for(int i = threadIdx.x; i < fnum; i += blockDim.x){
 					const float* v0 = vertices + faces[i * 3] * 3;
 					const float* v1 = vertices + faces[i * 3 + 1] * 3;
 					const float* v2 = vertices + faces[i * 3 + 2] * 3;

 					float v0v1[3]{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
 					float v0v2[3]{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};

 					float normal[3];
 					cross(v0v1, v0v2, normal);

 					for(int j = 0; j < 3; ++j){
 						int vidx = faces[i * 3  + j];
 						for(int k = 0; k < 3; ++k){
 							atomicAdd(vert_normal + vidx * 3 + k, normal[k]);
 						}
 					}
 				}
 			}

		""")

		self.func_cal_vnormal = mod.get_function("cal_vert_normal")
		self.func_norm_vnormal = mod.get_function("norm_vert_normal")
		self.func_cal_fnormal = mod.get_function("cal_face_normal")

	def cal_vert_normal(self):
		if not hasattr(self, 'func_norm_vnormal'):
			self.init_cuda_functions()

		self.vert_normal = transfer_data_to_gpu(np.zeros((self.vert_num, 3)).astype(np.float32))
		self.func_cal_vnormal(self.vertices, self.faces, self.vert_normal, \
			np.int32(self.vert_num), np.int32(self.face_num), \
			block = (1024, 1, 1))
		context.synchronize()

		self.func_norm_vnormal(self.vert_normal, np.int32(self.vert_num), \
			block = (1024, 1, 1))
		context.synchronize()

	def cal_face_normal(self):
		if not hasattr(self, 'func_cal_fnormal'):
			self.init_cuda_functions()
			
		self.face_normal = transfer_data_to_gpu(np.zeros((self.face_num, 3)).astype(np.float32))
		self.func_cal_fnormal(self.vertices, self.faces, self.face_normal, \
			np.int32(self.vert_num), np.int32(self.face_num), \
			block = (1024, 1, 1))
		context.synchronize()

	def init_mesh(self, mesh):
		self.vertices = transfer_data_to_gpu(mesh.vertices.astype(np.float32))
		if not mesh.faces is None:
			self.faces = transfer_data_to_gpu(mesh.faces.astype(np.int32))
			self.face_num = mesh.face_num()
		self.vert_num = mesh.vert_num()

		if hasattr(mesh, 'vert_normal') and not mesh.vert_normal is None:
			self.vert_normal = transfer_data_to_gpu(mesh.vert_normal.astype(np.float32))
