import numpy as np 


class Intersect:
	def __init__(self):
		pass

	def rays_tris_intersect(self, rays_start, rays_dir, tris):
		v0v1 = tris[:, 1, :] - tris[:, 0, :] 
		v0v2 = tris[:, 2, :] - tris[:, 0, :]

		pvec = np.cross(rays_dir, v0v2, axis = 1) 
		det = np.sum(v0v1 * pvec, axis = 1)
		check_ind = np.where(np.abs(det) > 1e-12)[0]
		inv_det = 1 / det[check_ind]

		u = np.full(len(rays_start), -1).astype(np.float32)
		v = np.full(len(rays_start), -1).astype(np.float32)
		t = np.full(len(rays_start), np.inf).astype(np.float32)

		tvec = rays_start[check_ind] - tris[check_ind, 0, :]
		u[check_ind] = np.sum(tvec * pvec[check_ind], axis = 1) * inv_det 

		qvec = np.cross(tvec, v0v1[check_ind])
		v[check_ind] = np.sum(qvec * rays_dir[check_ind], axis = 1) * inv_det 

		t[check_ind] = np.sum(v0v2[check_ind] * qvec, axis = 1) * inv_det

		return u, v, t

	def rays_aabb_intersect(self, rays_start, rays_dir, aabb_start, aabb_end, check_dir = False):
		dir_inv = np.zeros(rays_dir.shape, dtype = np.float32)
		dir_inv[np.abs(rays_dir) < 1e-10] = np.inf 
		dir_inv[np.abs(rays_dir) >= 1e-10] = np.divide(1.0, rays_dir[np.abs(rays_dir) >= 1e-10])

		t1 = (aabb_start - rays_start) * dir_inv
		t2 = (aabb_end - rays_start) * dir_inv 

		t = np.array([t1, t2])
		tmin = np.min(t, axis = 0)
		tmax = np.max(t, axis = 0)

		t_enter = np.max(tmin, axis = 1)
		t_exit = np.min(tmax, axis = 1)

		if check_dir:
			return np.logical_and(t_enter < t_exit, t_exit >= 0)
		return t_enter < t_exit

	def rays_octree_intersect(self, octree, vertices, faces, face_normals, rays_start, rays_dir, ray_ind, dist_threshold, normal_threshold):
		if not octree.item_ind is None:
			rays_num = len(ray_ind)
			tris_num = len(octree.item_ind)

			ray_ind_repeat = ray_ind.repeat(tris_num)
			u, v, t = self.rays_tris_intersect(rays_start[ray_ind_repeat],\
				rays_dir[ray_ind_repeat], \
				vertices[faces[np.tile(octree.item_ind, rays_num)]])

			u = u.reshape((rays_num, tris_num))
			v = v.reshape((rays_num, tris_num))
			t = t.reshape((rays_num, tris_num))

			intersect_mask = np.logical_and(np.logical_and(np.logical_and(u <= 1, u >= 0), np.logical_and(v <= 1, v >= 0)), u + v <= 1)
			t[np.where(~intersect_mask)] = np.inf
			t[np.where(t > dist_threshold)] = np.inf

			int_ray_ind, int_face_ind = np.where(t < np.inf - 1)
			dist = np.abs(t[int_ray_ind, int_face_ind])
			weights = np.column_stack(((1 - u - v)[int_ray_ind, int_face_ind], u[int_ray_ind, int_face_ind], v[int_ray_ind, int_face_ind]))
			int_ray_ind = ray_ind[int_ray_ind]
			int_face_ind = octree.item_ind[int_face_ind]

			cos = np.sum(rays_dir[int_ray_ind] * face_normals[int_face_ind], axis = 1)
			n_valid_mask = cos > normal_threshold

			return int_ray_ind[n_valid_mask], int_face_ind[n_valid_mask], weights[n_valid_mask].flatten(), dist[n_valid_mask].flatten() 
		else:
			int_ray_ind = np.array([], dtype = np.int32)
			int_face_ind = np.array([], dtype = np.int32)
			weights = np.array([], dtype = np.float32)
			dist = np.array([], dtype = np.float32)

			for i in range(len(octree.childs)):
				ray_check_mask = self.rays_aabb_intersect(rays_start[ray_ind], rays_dir[ray_ind], octree.childs[i].start, octree.childs[i].end)
				ray_check_ind = ray_ind[ray_check_mask]
				if len(ray_check_ind) == 0:
					continue 

				child_int_ray_ind, child_int_face_ind, child_weights, childs_dist = self.rays_octree_intersect(\
					octree.childs[i], \
					vertices, faces, face_normals, \
					rays_start, rays_dir, ray_check_ind, dist_threshold, normal_threshold)

				int_ray_ind = np.append(int_ray_ind, child_int_ray_ind)
				int_face_ind = np.append(int_face_ind, child_int_face_ind)
				weights = np.append(weights, child_weights)
				dist = np.append(dist, childs_dist)

			return int_ray_ind, int_face_ind, weights, dist

if __name__ == '__main__':
	import sys
	sys.path.append('.')
	from common.mesh import TriMesh
	root = 'C:/Users/wyt60/Desktop/tumvfr/input/st_6/'
	mesh = TriMesh()
	mesh.load(root + 'deform_lap_sphere.obj')
	mesh.cal_face_normal()

	from common.cor.octree import OCTree
	octree = OCTree()
	octree.from_triangles(mesh.vertices, mesh.faces, np.arange(mesh.faces.shape[0]))

	rays_o = np.loadtxt(root + 'rays_o.txt', dtype = np.float32)
	rays_v = np.loadtxt(root + 'rays_v.txt', dtype = np.float32)

	tester = Intersect()
	int_ray_ind, int_face_ind, weights, dist = tester.rays_octree_intersect(octree, mesh.vertices, mesh.faces, mesh.face_normal, \
		rays_o, rays_v, np.arange(len(rays_v)), 100, -1)

	pts = rays_o + dist[:, np.newaxis] * rays_v

	np.savetxt('./pts.txt', pts, fmt = '%5.5f')


import pycuda.driver as cuda 
from pycuda.autoinit import context
from pycuda.compiler import SourceModule

import struct
from common.utils import transfer_data_to_gpu

class Intersect_cuda:
	def __init__(self):
		mod = SourceModule("""
			#define INF __int_as_float(0x7f800000)
			#define INF_NEG __int_as_float(0xff800000)
			#include <stdint.h>

			__device__ void cross(
				const float* v1, const float* v2,
				float* v
			){
				v[0] = v1[1] * v2[2] - v2[1] * v1[2];
				v[1] = v1[2] * v2[0] - v2[2] * v1[0];
				v[2] = v1[0] * v2[1] - v2[0] * v1[1];
			}

			__device__ float dot(
				const float* v1, const float* v2, int dim
			){
				float res = 0;
				for(int i = 0; i < dim; ++i)
					res += v1[i] * v2[i];
				return res;
			}

			__device__ void normalize(
				const float* vec, float* nvec, int dim
			){
				float divide = 0;
				for(int i = 0; i < dim; ++i)
					divide += vec[i] * vec[i];
				divide = sqrt(divide);

				for(int i = 0; i < dim; ++i)
					nvec[i] = vec[i] / divide;
			}

			struct OCTree{
				OCTree *child_ptr[8];
				int child_num;
				float start[3];
				float end[3];
				int* ptr;
				int num;
			};

			extern "C++" {
				#define STACK_MAX_NUM 43
				template<typename T>
				struct Stack{
					T items[STACK_MAX_NUM];
					int top = -1;
				};

				template<typename T>
				__device__ void push(Stack<T>& s, T& oc){
					s.top++;
					s.items[s.top] = oc;
				}

				template<typename T>
				__device__ T pop(Stack<T>& s){
					T res = s.items[s.top];
					s.top--;
					return res;
				}

				template<typename T>
				__device__ bool isEmpty(Stack<T>& s){
					if(s.top == -1)
						return true;
					else
						return false;
				}
			}
			
			__device__ bool ray_tri_intersect(
				const float* v0, const float* v1, const float* v2, 
				const float* orig, const float* dir,
				float* res
			){
				float v0v1[3]{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
				float v0v2[3]{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};

				float pvec[3];
				cross(dir, v0v2, pvec);
				float det = dot(v0v1, pvec, 3);
				if(abs(det) < 1e-12) return false;

				float inv_det = 1.0f / det;
				float tvec[3]{orig[0] - v0[0], orig[1] - v0[1], orig[2] - v0[2]};
				res[0] = dot(tvec, pvec, 3) * inv_det;
				if(res[0] > 1 || res[0] < 0) return false;
				
				float qvec[3];
				cross(tvec, v0v1, qvec);
				res[1] = dot(dir, qvec, 3) * inv_det;
				if(res[1] < 0 || res[0] + res[1] >= 1) return false;

				res[2] = abs(dot(v0v2, qvec, 3) * inv_det);
				return true;
			}

			__device__ bool ray_start_in_aabb(
				const float* aabb_start, const float* aabb_end,
				const float* ray_start
			){
				return ray_start[0] >= aabb_start[0] && ray_start[0] <= aabb_end[0] && ray_start[1] >= aabb_start[1] && ray_start[1] <= aabb_end[1] && ray_start[2] >= aabb_start[2] && ray_start[2] <= aabb_end[2];
			}

			__device__ void cal_ray_dir_inv(
				const float *ray_dir, float *dir_inv
			){
				for(int i = 0; i < 3; ++i){
					if(ray_dir[i] > 0 && ray_dir[i] < 1e-12)
						dir_inv[i] = INF;
					else if(ray_dir[i] < 0 && ray_dir[i] > -1e-12)
						dir_inv[i] = INF_NEG;
					else
						dir_inv[i] = 1.0 / ray_dir[i];
				}
			}

			__device__ float ray_aabb_intersect(
				const float* aabb_start, const float* aabb_end,
				const float* ray_start, const float* dir_inv
			){
				if(ray_start_in_aabb(aabb_start, aabb_end, ray_start))
					return 0.0f;

				float t1 = (aabb_start[0] - ray_start[0]) * dir_inv[0];
				float t2 = (aabb_end[0] - ray_start[0]) * dir_inv[0];
				float t3 = (aabb_start[1] - ray_start[1]) * dir_inv[1];
				float t4 = (aabb_end[1] - ray_start[1]) * dir_inv[1];
				float t5 = (aabb_start[2] - ray_start[2]) * dir_inv[2];
				float t6 = (aabb_end[2] - ray_start[2]) * dir_inv[2];

				float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
				float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));
				if(tmin > tmax)
					return INF;
				else
					return tmin;
			}

			__device__ float point_2_plane_dist(
				const float* point, const float* p_plane, const float* normal
			){
				float vec[3];
				for(int i = 0; i < 3; ++i)
					vec[i] = point[i] - p_plane[i];
				return abs(dot(vec, normal, 3));
			}

			__global__ void rays_mesh_intersect(
				const float* vertices, const int* faces, const float* face_normals,
				const float* rays_start, const float* rays_dir, const float* dist_thres, const float* norm_thres,
				OCTree* octree,
				int* int_face_ind, float* int_weights, int ray_num
			){
				for(int i = threadIdx.x; i < ray_num; i += blockDim.x){
					float min_dist = dist_thres[i];

					float dir_inv[3];
					cal_ray_dir_inv(rays_dir + i * 3, dir_inv);

					float aabb_dist = ray_aabb_intersect(octree->start, octree->end, rays_start + i * 3, dir_inv);
					if(aabb_dist >= min_dist) continue;

					Stack<OCTree*> s;
					push(s, octree);
					while(!isEmpty(s)){
						OCTree *node = pop(s);	

						if(node->child_num > 0){
							for(int j = 0; j < node->child_num; ++j){
								aabb_dist = ray_aabb_intersect(node->child_ptr[j]->start, node->child_ptr[j]->end, rays_start + i * 3, dir_inv);
								if(aabb_dist < min_dist) push(s, node->child_ptr[j]);
							}
						}else{
							float res[3]{0, 0, 0};
							for(int j = 0; j < node->num; ++j){
								float cos_n = dot(rays_dir + i * 3, face_normals + node->ptr[j] * 3, 3);
								if(cos_n <= norm_thres[i])
									continue;

								const int* tri = faces + node->ptr[j] * 3;

								float dist = point_2_plane_dist(rays_start + i * 3, vertices + tri[0] * 3, face_normals + node->ptr[j] * 3);
								if(dist >= min_dist)
									continue;

								bool tri_int = ray_tri_intersect(vertices + tri[0] * 3, vertices + tri[1] * 3, vertices + tri[2] * 3, rays_start + i * 3, rays_dir + i * 3, res);
								if(!tri_int)
									continue;

								if(res[2] < min_dist){
									min_dist = res[2];
									int_face_ind[i] = node->ptr[j];
									int_weights[i * 3] = 1 - res[0] - res[1];
									int_weights[i * 3 + 1] = res[0];
									int_weights[i * 3 + 2] = res[1];
								}
							}
						}
					}
				}
			}

			""")
		self.func_rays_mesh = mod.get_function("rays_mesh_intersect")

	def rays_octree_intersect_cuda(self, octree_cuda, src_cuda, tgt_cuda, \
		dist_thres_lst, normal_thres_lst):

		int_face_ind = np.full(src_cuda.vert_num, -1, dtype = np.int32)
		int_face_ind_gpu = transfer_data_to_gpu(int_face_ind)
		int_weights = np.zeros((src_cuda.vert_num, 3), dtype = np.float32)
		int_weights_gpu = transfer_data_to_gpu(int_weights)

		dist_thres_lst_gpu = transfer_data_to_gpu(dist_thres_lst.astype(np.float32))
		normal_thres_lst_gpu = transfer_data_to_gpu(normal_thres_lst.astype(np.float32))

		self.func_rays_mesh(
			tgt_cuda.vertices, tgt_cuda.faces, tgt_cuda.face_normal, \
			src_cuda.vertices, src_cuda.vert_normal, dist_thres_lst_gpu, normal_thres_lst_gpu, \
			octree_cuda, \
			int_face_ind_gpu, int_weights_gpu, np.int32(src_cuda.vert_num), \
			block = (1024, 1, 1))

		context.synchronize()

		cuda.memcpy_dtoh(int_face_ind, int_face_ind_gpu)
		cuda.memcpy_dtoh(int_weights, int_weights_gpu)

		return int_face_ind, int_weights

