import numpy as np 
import cv2 as cv 

from common.utils import transfer_data_to_gpu

import pycuda.driver as cuda 
from pycuda.autoinit import context
from pycuda.compiler import SourceModule

class UVMapping:
	def __init__(self):
		pass

	def barycentric_coord(self, triangle, points):
		v0 = np.reshape(triangle[1] - triangle[0], (1, -1)) 
		v1 = np.reshape(triangle[2] - triangle[0], (1, -1))
		v2 = points - triangle[0]
		
		d00 = np.sum(v0 ** 2, axis = 1)
		d11 = np.sum(v1 ** 2, axis = 1)
		d01 = np.sum(v0 * v1, axis = 1)
		
		d20 = np.sum(v0 * v2, axis = 1)
		d21 = np.sum(v1 * v2, axis = 1)
		
		inv_denom = 1.0 / (d00 * d11 - d01 * d01) 
		
		v = (d11 * d20 - d01 * d21) * inv_denom
		w = (d00 * d21 - d01 * d20) * inv_denom
		u = 1 - v - w 
		return u.flatten(), v.flatten(), w.flatten()

	def map_uv_to_tri(self, tex_coords, faces_tc, tex_size): 
		mask = np.zeros((tex_size, tex_size), dtype = np.uint8)
		face_inds = np.zeros((tex_size, tex_size), dtype = np.int32)
		weights = np.zeros((tex_size, tex_size, 3), dtype = np.float32)

		for i in range(len(faces_tc)):
			triangle = tex_coords[faces_tc[i]] * tex_size

			bb_min = (np.min(triangle, axis = 0)).astype(np.int32)
			bb_max = np.ceil(np.max(triangle, axis = 0)).astype(np.int32)
			if bb_max[0] <= bb_min[0] or bb_max[1] <= bb_min[1]:
				continue

			x_ind = np.linspace(bb_min[0], bb_max[0], bb_max[0] - bb_min[0] + 1).astype(np.int32)
			y_ind = np.linspace(bb_min[1], bb_max[1], bb_max[1] - bb_min[1] + 1).astype(np.int32)

			points = np.column_stack((x_ind.repeat(len(y_ind)), np.tile(y_ind, len(x_ind))))
			u, v, w = self.barycentric_coord(triangle, points + 0.5)

			valid_ind = np.logical_and(np.logical_and(np.logical_and(u >= 0, u <= 1), np.logical_and(v >= 0, v <= 1)), u + v <= 1)
			if valid_ind.any() == False:
				continue

			coord_tri = points[valid_ind]

			mask[coord_tri[:, 1], coord_tri[:, 0]] = 1
			weights[coord_tri[:, 1], coord_tri[:, 0]] = np.column_stack((u[valid_ind], v[valid_ind], w[valid_ind]))
			face_inds[coord_tri[:, 1], coord_tri[:, 0]] = i

		return mask, face_inds, weights

class UVMapping_cuda:
	def __init__(self):
		mod = SourceModule("""
			#include <stdint.h>
			
			__global__ void map_uv_to_pos_map(
				const float* vertices, const int* faces,
				const uint8_t* mask, const int* face_inds_map, const float* weights_map,
				float* pos_map,
				int tex_size
			){
				for(int idx = threadIdx.x; idx < tex_size * tex_size; idx += blockDim.x){
					if(mask[idx] == 0)
						continue;
					int face_ind = face_inds_map[idx];
					int vert_ind[3]{faces[face_ind * 3], faces[face_ind * 3 + 1], faces[face_ind * 3 + 2]};

					for(int dim = 0; dim < 3; ++dim){
						pos_map[idx * 3 + dim] = 
						vertices[vert_ind[0] * 3 + dim] * weights_map[idx * 3] +
						vertices[vert_ind[1] * 3 + dim] * weights_map[idx * 3 + 1] +
						vertices[vert_ind[2] * 3 + dim] * weights_map[idx * 3 + 2];
					}			
				}
			}

			__global__ void map_uv_to_pos_vec(
				const float* vertices, const int* faces, 
				const int* row_ind, const int* col_ind, const int* face_inds_map, const float* weights_map,
				float* pos_vec, int vnum, int tex_size
			){
				for(int i = threadIdx.x; i < vnum; i += blockDim.x){
					int idx = row_ind[i] * tex_size + col_ind[i];
					int face_ind = face_inds_map[idx];
					int vert_ind[3]{faces[face_ind * 3], faces[face_ind * 3 + 1], faces[face_ind * 3 + 2]};
					for(int dim = 0; dim < 3; ++dim){
						pos_vec[i * 3 + dim] = 
						vertices[vert_ind[0] * 3 + dim] * weights_map[idx * 3] +
						vertices[vert_ind[1] * 3 + dim] * weights_map[idx * 3 + 1] + 
						vertices[vert_ind[2] * 3 + dim] * weights_map[idx * 3 + 2];
					}
				}
			}

			__device__ void barycentric_coord(
				const float* tri,
				const float* p,
				float* weight, int dim
			){
				float v0[3], v1[3], v2[3];
				for(int i = 0; i < dim; ++i){
					v0[i] = tri[i + dim] - tri[i];
					v1[i] = tri[i + dim * 2] - tri[i];
					v2[i] = p[i] - tri[i];
				}

				float d00 = 0, d11 = 0, d01 = 0, d20 = 0, d21 = 0;
				for(int i = 0; i < dim; ++i){
					d00 += v0[i] * v0[i];
					d11 += v1[i] * v1[i];
					d01 += v0[i] * v1[i];
					d20 += v0[i] * v2[i];
					d21 += v1[i] * v2[i];
				}

				float inv_denom = 1.0 / (d00 * d11 - d01 * d01);
				weight[1] = (d11 * d20 - d01 * d21) * inv_denom;
				weight[2] = (d00 * d21 - d01 * d20) * inv_denom;
				weight[0] = 1 - weight[1] - weight[2];
			}
			
			__global__ void map_uv_to_tri(
				const float* tex_coords, const int* faces_tc, 
				int tc_num, int face_num, int tex_size, 
				uint8_t* mask, int* face_inds_map, float* weights_map
			){
				for(int t = threadIdx.x; t < face_num; t += blockDim.x){
					float triangle[6];
					int bb_min[2]{tex_size + 100, tex_size + 100};
					int bb_max[2]{-1, -1};
					for(int i = 0; i < 3; ++i){
						triangle[i * 2] = tex_coords[faces_tc[t * 3 + i] * 2] * tex_size;
						triangle[i * 2 + 1] = tex_coords[faces_tc[t * 3 + i] * 2 + 1] * tex_size;

						bb_min[0] = bb_min[0] > floorf(triangle[i * 2]) ? floorf(triangle[i * 2]) : bb_min[0];
						bb_min[1] = bb_min[1] > floorf(triangle[i * 2 + 1]) ? floorf(triangle[i * 2 + 1]) : bb_min[1];
						bb_max[0] = bb_max[0] < ceilf(triangle[i * 2]) ? ceilf(triangle[i * 2]) : bb_max[0];
						bb_max[1] = bb_max[1] < ceilf(triangle[i * 2 + 1]) ? ceilf(triangle[i * 2 + 1]) : bb_max[1]; 
					}

					bb_min[0] = bb_min[0] >= 0 ? bb_min[0] : 0;
					bb_min[1] = bb_min[1] >= 0 ? bb_min[1] : 0;
					bb_max[0] = bb_max[0] < tex_size ? bb_max[0] : tex_size - 1;
					bb_max[1] = bb_max[1] < tex_size ? bb_max[1] : tex_size - 1 ;

					for(int j = bb_min[1]; j < bb_max[1] + 1; ++j){
						for(int i = bb_min[0]; i < bb_max[0] + 1; ++i){
							float weight[3]{0.0f, 0.0f, 0.0f};
							float point[2]{i + 0.5f, j + 0.5f};
							barycentric_coord(triangle, point, weight, 2);

							if(weight[0] <= 1 && weight[0] >= 0 && weight[1] <= 1 && weight[1] >= 0 && weight[2] <= 1 && weight[2] >= 0){
								mask[j * tex_size + i] = 1;
								face_inds_map[j * tex_size + i] = (int)t;
								for(int k = 0; k < 3; ++k){
									weights_map[(j * tex_size + i) * 3 + k] =  weight[k];
								}
							}
						}
					}
				}
			}

			""")
		self.func_map_uv_to_tri = mod.get_function("map_uv_to_tri")
		self.func_map_uv_to_pos_map = mod.get_function("map_uv_to_pos_map")
		self.func_map_uv_to_pos_vec = mod.get_function("map_uv_to_pos_vec")

	def map_uv_to_tri_gpu(self, tex_coords, faces_tc, tex_size):
		tc_gpu = transfer_data_to_gpu(tex_coords.astype(np.float32))
		faces_tc_gpu = transfer_data_to_gpu(faces_tc.astype(np.int32))

		mask_gpu = cuda.mem_alloc(tex_size * tex_size)
		face_inds_gpu = cuda.mem_alloc(tex_size * tex_size * 4)
		weights_gpu = cuda.mem_alloc(tex_size * tex_size * 3 * 4)

		self.func_map_uv_to_tri(tc_gpu, faces_tc_gpu, \
			np.int32(len(tex_coords)), np.int32(len(faces_tc)), np.int32(tex_size), \
			mask_gpu, face_inds_gpu, weights_gpu, block = (1024, 1, 1))

		context.synchronize()
		return mask_gpu, face_inds_gpu, weights_gpu

	def map_uv_to_pos_map_gpu(self, verts_gpu, faces_gpu, mask_gpu, face_inds_map_gpu, weights_map_gpu, tex_size):
		pos_map_gpu = cuda.mem_alloc(tex_size * tex_size * 4 * 3)
		self.func_map_uv_to_pos_map(verts_gpu, faces_gpu, \
			mask_gpu, face_inds_map_gpu, weights_map_gpu, \
			pos_map_gpu, np.int32(tex_size),\
			block = (1024, 1, 1))
		context.synchronize()

		pos_map = np.zeros((tex_size, tex_size, 3), dtype = np.float32)
		cuda.memcpy_dtoh(pos_map, pos_map_gpu)
		return pos_map

	def map_uv_to_pos_vec_gpu(self, verts_gpu, faces_gpu, \
		row_ind_gpu, col_ind_gpu, face_inds_map_gpu, weights_map_gpu, \
		tex_size, vnum):
		pos_vec_gpu = cuda.mem_alloc(vnum * 3 * 4)
		self.func_map_uv_to_pos_vec(verts_gpu, faces_gpu, \
			row_ind_gpu, col_ind_gpu, face_inds_map_gpu, weights_map_gpu, \
			pos_vec_gpu, np.int32(vnum), np.int32(tex_size),\
			block = (1024, 1, 1))
		context.synchronize()

		return pos_vec_gpu