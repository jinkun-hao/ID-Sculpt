import numpy as np 
import scipy.sparse as sp
from scipy.sparse import linalg as spl
import cv2 as cv 

import pycuda.driver as cuda 
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

from common.utils import transfer_data_to_gpu

class PoissonEdit:
	def __init__(self, use_cuda = True):
		if not use_cuda:
			return 
		mod = SourceModule("""
			#include <stdint.h>
			__global__ void construct_b(
				const uint8_t* src, const int16_t* u, const int16_t* v,
				float* b,
				int pix_num, int size
			){
				for(int i = threadIdx.x; i < pix_num; i += blockDim.x){
					int cnt = 0;
					int16_t u_lst[4]{u[i], u[i], u[i] + 1, u[i] - 1};
					int16_t v_lst[4]{v[i] - 1, v[i] + 1, v[i], v[i]};

					for(int j = 0; j < 4; ++j){
						if(u_lst[j] >= 0 && u_lst[j] <= size - 1 && v_lst[j] >= 0 && v_lst[j] <= size - 1){
							cnt += 1;
							for(int k = 0; k < 3; ++k)
								b[i * 3 + k] -= float(src[(v_lst[j] * size + u_lst[j]) * 3 + k]);
						}
					}
					for(int k = 0; k < 3; ++k)
						b[i * 3 + k] += float(cnt) * float(src[(v[i] * size + u[i]) * 3 + k]);
				}
			}
			__global__ void sor_iter(
				const uint8_t* tgt, const float* b,
				const int* uv_to_ind, const int16_t* u, const int16_t* v,
				float* x, float* err, 
				int pix_num, int size, int16_t red, float w
			){
				for(int i = threadIdx.x; i < pix_num; i += blockDim.x){
					if(red != (u[i] + v[i]) % 2)
						continue;

					float res[3]{b[i * 3], b[i * 3 + 1], b[i * 3 + 2]};
					int16_t u_lst[4]{u[i], u[i], u[i] + 1, u[i] - 1};
					int16_t v_lst[4]{v[i] - 1, v[i] + 1, v[i], v[i]};

					int cnt = 0;

					for(int j = 0; j < 4; ++j){
						if(u_lst[j] >= 0 && u_lst[j] <= size - 1 && v_lst[j] >= 0 && v_lst[j] <= size - 1){
							cnt += 1;
							if(uv_to_ind[v_lst[j] * size + u_lst[j]] >= 0){
								for(int k = 0; k < 3; ++k)
									res[k] += x[uv_to_ind[v_lst[j] * size + u_lst[j]] * 3 + k];
							}else{
								for(int k = 0; k < 3; ++k)
									res[k] += float(tgt[(v_lst[j] * size + u_lst[j]) * 3 + k]);
							}
						}
					}

					err[i] = 0;
					for(int k = 0; k < 3; ++k){
						float num = res[k] * w / float(cnt) + (1 - w) * x[i * 3 + k];
						err[i] += abs(num - x[i * 3 + k]) / 3;
						x[i * 3 + k] = num;
						//x[i * 3 + k] = res[k] / float(cnt); gauss seidel iteration
					}
				}
			}
			""")
		self.func_construct = mod.get_function("construct_b")
		self.func_solve = mod.get_function("sor_iter")

	def poisson_mat_sparse(self, src, tgt, mask):
		src = src.astype(np.float32)
		tgt = tgt.astype(np.float32)
		height, width = tgt.shape[:2]

		uv_to_ind = -1 * np.ones((height, width))
		v, u = np.where(mask)
		uv_to_ind[v, u] = np.arange(len(u), dtype = np.int32)

		data, row, col = [], [], []
		b = np.zeros((len(u), 3), dtype = np.float32)

		for i in range(len(u)):
			cnt = 0
			u_lst = [u[i], u[i], u[i] - 1, u[i] + 1]
			v_lst = [v[i] - 1, v[i] + 1, v[i], v[i]]

			for j in range(4):
				if u_lst[j] >= 0 and v_lst[j] >= 0 and u_lst[j] <= width - 1 and v_lst[j] <= height - 1:
					b[i] -= src[v_lst[j], u_lst[j]]
					cnt += 1 
					if mask[v_lst[j], u_lst[j]] > 0:
						data.append(-1)
						row.append(i)
						col.append(uv_to_ind[v_lst[j], u_lst[j]])
					else:
						b[i] += tgt[v_lst[j], u_lst[j]]

			data.append(cnt)
			row.append(i)
			col.append(i)

			b[i] += cnt * src[v[i], u[i]]

		A = sp.coo_matrix((data, (row, col)), shape = (len(u), len(u)))
		A.sum_duplicates()
		return A, b

	def blend(self, src, tgt, mask):
		A, b = self.poisson_mat_sparse(src, tgt, mask)
		x_r = spl.cg(A, b[:, 0], x0 = src[np.where(mask)][:, 0])[0]
		x_g = spl.cg(A, b[:, 1], x0 = src[np.where(mask)][:, 1])[0]
		x_b = spl.cg(A, b[:, 2], x0 = src[np.where(mask)][:, 2])[0]

		res = tgt.copy()
		v, u = np.where(mask)
		res[v, u] = np.clip(np.column_stack((x_r, x_g, x_b)), 0, 255).astype(np.uint8)

		return res 

	def blend_gs_cuda(self, src, tgt, mask, ini = None, max_iters = 1000, threshold = 0.02):
		size = src.shape[0]
		src_gpu = transfer_data_to_gpu(src.astype(np.uint8))
		tgt_gpu = transfer_data_to_gpu(tgt.astype(np.uint8))

		v, u = np.where(mask)
		v_gpu = transfer_data_to_gpu(v.astype(np.int16))
		u_gpu = transfer_data_to_gpu(u.astype(np.int16))
		uv_to_ind = np.full((size, size), -1, dtype = np.int32)
		uv_to_ind[v, u] = np.arange(len(u), dtype = np.int32)
		uv_to_ind_gpu = transfer_data_to_gpu(uv_to_ind)

		b = np.zeros((len(u), 3), dtype = np.float32)
		b_gpu = transfer_data_to_gpu(b)
		self.func_construct(src_gpu, u_gpu, v_gpu, \
			b_gpu, np.int32(len(u)), np.int32(size), \
			block = (1024, 1, 1))
		context.synchronize()

		cuda.memcpy_dtoh(b, b_gpu)

		if ini is None:
			x = src[v, u].astype(np.float32)
		else:
			x = ini[v, u].astype(np.float32)
		x_gpu = transfer_data_to_gpu(x)

		err = np.zeros(len(u), dtype = np.float32)
		err_gpu = transfer_data_to_gpu(err)

		for i in range(max_iters):
			self.func_solve(tgt_gpu, b_gpu, \
				uv_to_ind_gpu, u_gpu, v_gpu, \
				x_gpu, err_gpu, np.int32(len(u)), np.int32(size), np.int16(1), np.float32(1.95),\
				block = (1024, 1, 1))
			context.synchronize()

			self.func_solve(tgt_gpu, b_gpu, \
				uv_to_ind_gpu, u_gpu, v_gpu, \
				x_gpu, err_gpu, np.int32(len(u)), np.int32(size), np.int16(0), np.float32(1.95),\
				block = (1024, 1, 1))
			context.synchronize()
			
			cuda.memcpy_dtoh(err, err_gpu)
			p_err = np.max(err)
			if p_err < threshold:
				break
		print('poisson iters:', i)

		cuda.memcpy_dtoh(x, x_gpu)

		res = tgt.copy()
		res[v, u] = np.clip(x, 0, 255).astype(np.uint8)

		return res

