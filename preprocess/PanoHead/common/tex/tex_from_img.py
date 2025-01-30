import numpy as np 
import cv2 as cv 
import torch

from common.mesh import TriMesh 
from common.utils import transfer_data_to_gpu

import pycuda.driver as cuda 
from pycuda.autoinit import context
from pycuda.compiler import SourceModule

class TexExtract:
	def __init__(self, tex_size = 512):
		self.tex_size = tex_size

	def init_cuda_func(self):
		mod = SourceModule("""
			#include <stdint.h>
			#define INF __int_as_float(0x7f800000)
			#define INF_NEG __int_as_float(0xff800000)
			#define PI 3.14159265357
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

			__device__ void normalize_vec(const float* vec, float* vec_norm, int dim){
				float num_sum = 0;
				for(int i = 0; i < dim; ++i)
					num_sum = vec[i] * vec[i];
				num_sum = sqrt(num_sum);
				for(int i = 0; i < dim; ++i)
					vec_norm[i] = vec[i] / num_sum;
			}

			__device__ void project_pers_atom(
				const float* vertex, const float* cam_mat,
				float* vert_proj
			){
				float w = vertex[0] * cam_mat[6] + vertex[1] * cam_mat[7] + vertex[2] * cam_mat[8];
				vert_proj[0] = (vertex[0] * cam_mat[0] + vertex[1] * cam_mat[1] + vertex[2] * cam_mat[2]) / w;
				vert_proj[1] = (vertex[0] * cam_mat[3] + vertex[1] * cam_mat[4] + vertex[2] * cam_mat[5]) / w;
			}

			__device__ void tri_2d_bbox(
				const float* triangle, int* bb_min, int* bb_max, int width, int height
			){
				bb_min[0] = width + 10, bb_min[1] = height + 10;
				bb_max[0] = -10, bb_max[1] = -10;
				for(int i = 0; i < 3; ++i){
					bb_min[0] = bb_min[0] > floor(triangle[i * 2]) ? floor(triangle[i * 2]) : bb_min[0];
					bb_min[1] = bb_min[1] > floor(triangle[i * 2 + 1]) ? floor(triangle[i * 2 + 1]) : bb_min[1];
					bb_max[0] = bb_max[0] < ceil(triangle[i * 2]) ? ceil(triangle[i * 2]) : bb_max[0];
					bb_max[1] = bb_max[1] < ceil(triangle[i * 2 + 1]) ? ceil(triangle[i * 2 + 1]) : bb_max[1]; 
				}

				bb_min[0] = bb_min[0] >= 0 ? bb_min[0] : 0;
				bb_min[1] = bb_min[1] >= 0 ? bb_min[1] : 0;
				bb_max[0] = bb_max[0] < width ? bb_max[0] : width - 1;
				bb_max[1] = bb_max[1] < height ? bb_max[1] : height - 1;
			}

			template<typename T>
			__device__ void bilinear_interp(
				const T* data, float* val,
				int w, int h, float u, float v, int dim
			){
				int u0 = (int)u;
				int u1 = u0 + 1;
				int v0 = (int)v;
				int v1 = v0 + 1;

				float wu0 = u1 - u, wu1 = u - u0;
				float wv0 = v1 - v, wv1 = v - v0;

				for(int i = 0; i < dim; ++i){
					float d00 = float(data[dim * (u0 + v0 * w) + i]);
					float d01 = float(data[dim * (u0 + v1 * w) + i]);
					float d10 = float(data[dim * (u1 + v0 * w) + i]);
					float d11 = float(data[dim * (u1 + v1 * w) + i]);

					val[i] = wu0 * (d00 * wv0 + d01 * wv1) + wu1 * (d10 * wv0 + d11 * wv1);
				}
			}

			extern "C"{
			__global__ void project_pers(
				const float* vertices, const float* cam_mat,
				float* vert_proj, int vert_num
			){
				for(int t = threadIdx.x; t < vert_num; t += blockDim.x)
					project_pers_atom(vertices + t * 3, cam_mat, vert_proj + t * 2);
			}

			__global__ void transform(
				const float* vertices, const float* trans_mat, 
				float* vert_trans,
				int vert_num
			){
				for(int t = threadIdx.x; t < vert_num; t += blockDim.x){
					vert_trans[t * 3 + 0] = vertices[t * 3] * trans_mat[0] + vertices[t * 3 + 1] * trans_mat[1] + vertices[t * 3 + 2] * trans_mat[2] + trans_mat[3];
					vert_trans[t * 3 + 1] = vertices[t * 3] * trans_mat[4] + vertices[t * 3 + 1] * trans_mat[5] + vertices[t * 3 + 2] * trans_mat[6] + trans_mat[7];
					vert_trans[t * 3 + 2] = vertices[t * 3] * trans_mat[8] + vertices[t * 3 + 1] * trans_mat[9] + vertices[t * 3 + 2] * trans_mat[10] + trans_mat[11];
				}
			}

			__global__ void zbuffer(
				const float* vert_trans, const float* vert_proj, 
				const uint16_t* faces,
				float* depth, uint8_t* mask,
				int vert_num, int face_num, int width, int height
			){
				for(int t = threadIdx.x; t < face_num; t += blockDim.x){
					float tri_proj[6];
					for(int i = 0; i < 3; ++i){
						tri_proj[i * 2] = vert_proj[faces[t * 3 + i] * 2];
						tri_proj[i * 2 + 1] = vert_proj[faces[t * 3 + i] * 2 + 1];
					}

					int bb_min[2]{0, 0}, bb_max[2]{0, 0};
					tri_2d_bbox(tri_proj, bb_min, bb_max, width, height);

					for(int j = bb_min[1]; j < bb_max[1] + 1; ++j){
						for(int i = bb_min[0]; i < bb_max[0] + 1; ++i){
							float weight[3]{0.0f, 0.0f, 0.0f};
							float point[2]{i + 0.5f, j + 0.5f};
							barycentric_coord(tri_proj, point, weight, 2);

							if(weight[0] <= 1 && weight[0] >= 0 && weight[1] <= 1 && weight[1] >= 0 && weight[2] <= 1 && weight[2] >= 0){
								mask[j * width + i] = 1;
								float d = 1 / (weight[0] / vert_trans[faces[t * 3] * 3 + 2] + weight[1] / vert_trans[faces[t * 3 + 1] * 3 + 2] + weight[2] / vert_trans[faces[t * 3 + 2] * 3 + 2]);
								depth[j * width + i] = depth[j * width + i] > d ? d : depth[j * width + i];
							}
						}
					}
				}
			}

			__global__ void extract_tex_occ(
				const float* vert_trans, const uint16_t* faces, const float* vert_normal,
				const float* tex_coords, const uint16_t* faces_tc, 
				const float* zbuffer, const uint8_t* img,
				const float* cam_mat,
				uint8_t* texture, uint8_t* mask,
				int face_num, int tex_size, int width, int height
			){
				for(int t = threadIdx.x; t < face_num; t += blockDim.x){
					float triangle[6];
					for(int i = 0; i < 3; ++i){
						triangle[i * 2] = tex_coords[faces_tc[t * 3 + i] * 2] * tex_size;
						triangle[i * 2 + 1] = tex_coords[faces_tc[t * 3 + i] * 2 + 1] * tex_size;
					}

					int bb_min[2]{0, 0}, bb_max[2]{0, 0};
					tri_2d_bbox(triangle, bb_min, bb_max, tex_size, tex_size);

					for(int j = bb_min[1]; j < bb_max[1] + 1; ++j){
						for(int i = bb_min[0]; i < bb_max[0] + 1; ++i){
							float weight[3]{0.0f, 0.0f, 0.0f};
							float point[2]{i + 0.5f, j + 0.5f};
							barycentric_coord(triangle, point, weight, 2);

							if(weight[0] > 1 && weight[0] < 0 && weight[1] > 1 && weight[1] < 0 && weight[2] > 1 && weight[2] < 0)
								continue;
							
							float pos[3]{0, 0, 0};
							float normal[3]{0, 0, 0};
							for(int m = 0; m < 3; ++m){
								for(int n = 0; n < 3; ++n){
									pos[m] += weight[n] * vert_trans[faces[t * 3 + n] * 3 + m];
									normal[m] += weight[n] * vert_normal[faces[t * 3 + n] * 3 + m];
								}
							}

							float proj[2]{0, 0};
							project_pers_atom(pos, cam_mat, proj);
							if(proj[0] < 0.5 || proj[0] > width || proj[1] < 0.5 || proj[1] > height)
								continue;

							float ray_vec[3]{0, 0, 0};
							normalize_vec(pos, ray_vec, 3);
							float norm_vec[3]{0, 0, 0};
							normalize_vec(normal, norm_vec, 3);
							float cosine = -(norm_vec[0] * ray_vec[0] + norm_vec[1] * ray_vec[1] + norm_vec[2] * ray_vec[2]);
							if(cosine < 0.2)
								continue;					

							float d;
							bilinear_interp<float>(zbuffer, &d, width, height, proj[0] - 0.5, proj[1] - 0.5, 1);
							if(d < pos[2] - 0.005)
								continue;

							float color[3]{0, 0, 0};
							bilinear_interp<uint8_t>(img, color, width, height, proj[0] - 0.5, proj[1] - 0.5, 3);
							for(int n = 0; n < 3; ++n){
								if(color[n] < 0) color[n] = 0;
								if(color[n] > 255) color[n] = 255;
								texture[(j * tex_size + i) * 3 + n] = uint8_t(color[n]);
							}
							
							mask[j * tex_size + i] = 1;
						}
					}
				}
			}
			}
			""", no_extern_c = 1)
		self.func_ext = mod.get_function("extract_tex_occ")
		self.func_trans = mod.get_function("transform")
		self.func_proj = mod.get_function("project_pers")
		self.func_zbuffer = mod.get_function("zbuffer")

	def extract_tex_from_img(self, img, mesh, trans_mat, cam_mat):
		if not hasattr(self, 'func_ext'):
			self.init_cuda_func()
		if not hasattr(mesh, 'vert_normal') or mesh.vert_normal is None:
			mesh.cal_vert_normal()

		height, width = img.shape[:2]
		height, width = np.int32(height), np.int32(width)
		vert_num = np.int32(mesh.vert_num())
		face_num = np.int32(mesh.face_num())

		img_gpu = transfer_data_to_gpu(img.astype(np.uint8))

		trans_mat_gpu = transfer_data_to_gpu(trans_mat.astype(np.float32))
		cam_mat_gpu = transfer_data_to_gpu(cam_mat.astype(np.float32))

		vertices_gpu = transfer_data_to_gpu(mesh.vertices.astype(np.float32))
		normal_gpu = transfer_data_to_gpu(mesh.vert_normal.astype(np.float32))
		tc_gpu = transfer_data_to_gpu(mesh.tex_coords.astype(np.float32))
		faces_gpu = transfer_data_to_gpu(mesh.faces.astype(np.uint16))
		if mesh.faces_tc is None:
			faces_tc_gpu = transfer_data_to_gpu(mesh.faces.astype(np.uint16)) 
		else:
			faces_tc_gpu = transfer_data_to_gpu(mesh.faces_tc.astype(np.uint16))

		vert_trans_gpu = cuda.mem_alloc(mesh.vert_num() * 3 * 4)
		self.func_trans(vertices_gpu, trans_mat_gpu, vert_trans_gpu, vert_num, block = (1024, 1, 1))
		context.synchronize()

		tmat_normal = trans_mat.copy()
		tmat_normal[:, 3] = 0 
		tmat_normal_gpu = transfer_data_to_gpu(tmat_normal.astype(np.float32))
		normal_trans_gpu =  cuda.mem_alloc(mesh.vert_num() * 3 * 4)
		self.func_trans(normal_gpu, tmat_normal_gpu, normal_trans_gpu, vert_num, block = (1024, 1, 1))
		context.synchronize()

		vert_proj_gpu = cuda.mem_alloc(mesh.vert_num() * 2 * 4)
		self.func_proj(vert_trans_gpu, cam_mat_gpu, vert_proj_gpu, vert_num, block = (1024, 1, 1))
		context.synchronize()

		zbuffer = np.full((img.shape[0], img.shape[1]), np.inf, dtype = np.float32)
		zbuffer_gpu = transfer_data_to_gpu(zbuffer)
		zmask_gpu = cuda.mem_alloc(img.shape[0] * img.shape[1])
		self.func_zbuffer(vert_trans_gpu, vert_proj_gpu, faces_gpu, zbuffer_gpu, zmask_gpu, \
			vert_num, face_num, width, height, \
			block = (1024, 1, 1))
		context.synchronize()

		tex_gpu = cuda.mem_alloc(self.tex_size * self.tex_size * 3)
		mask_gpu = cuda.mem_alloc(self.tex_size * self.tex_size)
		self.func_ext(vert_trans_gpu, faces_gpu, normal_trans_gpu, \
			tc_gpu, faces_tc_gpu, zbuffer_gpu, img_gpu, cam_mat_gpu, \
			tex_gpu, mask_gpu, face_num, np.int32(self.tex_size), width, height, \
			block = (1024, 1, 1))
		context.synchronize()

		texture = np.zeros((self.tex_size, self.tex_size, 3), dtype = np.uint8)
		mask = np.zeros((self.tex_size, self.tex_size), dtype = np.uint8)
		cuda.memcpy_dtoh(texture, tex_gpu)
		cuda.memcpy_dtoh(mask, mask_gpu)

		return texture, mask
