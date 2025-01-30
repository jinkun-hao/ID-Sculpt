import numpy as np 
import cv2 as cv 

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import *
from OpenGL.GLU import *
import glfw

import pycuda.driver as cuda 
from pycuda.autoinit import context
import pycuda.gl as cgl 
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gp

from common.mesh import TriMesh
from common.render.shader import Shader

class BS_Renderer:
	def __init__(self, width = 1000, height = 1000):
		self.width = width 
		self.height = height 

		self.init_cuda_functions()
		self.init_gl()

	def init_gl(self):
		if not glfw.init():
			return
		glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
		glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
		glfw.window_hint(glfw.SAMPLES, 4)
		glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

		self.window = glfw.create_window(self.width, self.height, "opengl", None, None)

		if not self.window:
			glfw.terminate()
			print('---init window fail---')
			return

		glfw.make_context_current(self.window)

		self.shader_tex = Shader('texture', self.width, self.height)
		self.shader_color = Shader('vert_color', self.width, self.height)
		
		glEnable(GL_MULTISAMPLE)
		glEnable(GL_DEPTH_TEST)

	def upload_tex(self, texture):
		obj_tex_id = glGenTextures(1)

		glBindTexture(GL_TEXTURE_2D, obj_tex_id)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture.shape[1], texture.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, texture)

		return obj_tex_id

	def init_cuda_functions(self):
		mod = SourceModule("""
			#include <stdint.h>
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

			__global__ void interp_bs(
				const float* bs, const float* coef,
				float* vertices, int vert_num, int coef_num
			){
				for(int i = threadIdx.x; i < vert_num; i += blockDim.x){
					vertices[i * 3] = bs[i * 3];
					vertices[i * 3 + 1] = bs[i * 3 + 1];
					vertices[i * 3 + 2] = bs[i * 3 + 2];

					for(int j = 1; j < coef_num; j++){
						vertices[i * 3] += coef[j - 1] * (bs[j * vert_num * 3 + i * 3] - bs[i * 3]);
						vertices[i * 3 + 1] += coef[j - 1] * (bs[j * vert_num * 3 + i * 3 + 1] - bs[i * 3 + 1]);
						vertices[i * 3 + 2] += coef[j - 1] * (bs[j * vert_num * 3 + i * 3 + 2] - bs[i * 3 + 2]);
					}
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

			__global__ void norm_vert_normal(float* vn, int vnum){
				for(int i = threadIdx.x; i < vnum; i += blockDim.x){
					normalize(vn + i * 3, 3);
				}
			}

			__global__ void arange_faces(
				const float* src, const int* faces,
				float* tgt,
				int vnum, int fnum, int dim
			){
				for(int i = threadIdx.x; i < fnum; i += blockDim.x){
					for(int j = 0; j < 3; j++){
						int vidx = faces[i * 3 + j];
						for(int k = 0; k < dim; k++){
							tgt[i * 3 * dim + j * dim + k] = src[vidx * dim + k];
						}
					}
				}
			}
			""")
		self.func_interp_bs = mod.get_function("interp_bs")
		self.func_cal_vn = mod.get_function("cal_vert_normal")
		self.func_norm_vn = mod.get_function("norm_vert_normal")
		self.func_arange = mod.get_function("arange_faces")

	def mem_alloc_gl_gpu(self, size, dtype = np.float32):
		data = np.zeros(size, dtype = np.uint8)

		pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)

		pbo_gpu = cgl.RegisteredBuffer(int(pbo))

		return pbo, pbo_gpu

	def move_to_gpu(self, data):
		data_gpu = cuda.mem_alloc(data.nbytes)
		cuda.memcpy_htod(data_gpu, data)
		return data_gpu

	def upload_object(self, bs, faces, use_tex = True, faces_tc = None, tex_coords = None, tex = None):
		op = {}
		op['vnum'] = len(bs[0])
		op['fnum'] = len(faces)
		op['cnum'] = len(bs)

		op['bs_gpu'] = self.move_to_gpu(bs.astype(np.float32))
		op['faces_gpu'] = self.move_to_gpu(faces.astype(np.int32))

		op['v_gpu_'] = self.move_to_gpu(np.zeros((op['vnum'], 3), dtype = np.float32))
		op['vn_gpu_'] = self.move_to_gpu(np.zeros((op['vnum'], 3), dtype = np.float32))

		op['v_bo'], op['v_gpu'] = self.mem_alloc_gl_gpu(op['fnum'] * 3 * 3 * 4)
		v_map = op['v_gpu'].map()
		op['v_pointer'], size = v_map.device_ptr_and_size()
		v_map.unmap()

		op['vn_bo'], op['vn_gpu'] = self.mem_alloc_gl_gpu(op['fnum'] * 3 * 3 * 4)
		vn_map = op['vn_gpu'].map()
		op['vn_pointer'], size = vn_map.device_ptr_and_size()
		vn_map.unmap()

		op['vao'] = glGenVertexArrays(1)
		glBindVertexArray(op['vao'])

		glBindBuffer(GL_ARRAY_BUFFER, op['v_bo'])
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
		glEnableVertexAttribArray(0)

		glBindBuffer(GL_ARRAY_BUFFER, op['vn_bo'])
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
		glEnableVertexAttribArray(1)

		if not use_tex:
			vert_color = np.full((op['vnum'], 3), 0.83).astype(np.float32)
			op['vc_bo'] = glGenBuffers(1)
			glBindBuffer(GL_ARRAY_BUFFER, op['vc_bo'])
			glBufferData(GL_ARRAY_BUFFER, op['fnum'] * 3 * 3 * 4, vert_color[faces].astype(np.float32), GL_STATIC_DRAW)

			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
			glEnableVertexAttribArray(2)
		else:
			op['tc_bo'] = glGenBuffers(1)
			glBindBuffer(GL_ARRAY_BUFFER, op['tc_bo'])
			glBufferData(GL_ARRAY_BUFFER, op['fnum'] * 3 * 2 * 4, tex_coords[faces_tc].astype(np.float32), GL_STATIC_DRAW)

			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
			glEnableVertexAttribArray(2)

			op['tbo'] = self.upload_tex(tex)

		glBindBuffer(GL_ARRAY_BUFFER, 0)
		glBindVertexArray(0)

		return op

	def interp_bs_object(self, op, coef_gpu):
		self.func_interp_bs(op['bs_gpu'], coef_gpu, op['v_gpu_'], \
			np.int32(op['vnum']), np.int32(op['cnum']), block = (1024, 1, 1))
		context.synchronize()

		self.func_cal_vn(op['v_gpu_'], op['faces_gpu'], op['vn_gpu_'], np.int32(op['vnum']), np.int32(op['fnum']), \
			block = (1024, 1, 1))
		context.synchronize()

		self.func_norm_vn(op['vn_gpu_'], np.int32(op['vnum']), \
			block = (1024, 1, 1))
		context.synchronize()

		self.func_arange(op['v_gpu_'], op['faces_gpu'], gp.GPUArray(op['fnum'] * 3 * 3, np.float32, gpudata = op['v_pointer']), \
			np.int32(op['vnum']), np.int32(op['fnum']), np.int32(3), \
			block = (1024, 1, 1))
		context.synchronize()

		self.func_arange(op['vn_gpu_'], op['faces_gpu'], gp.GPUArray(op['fnum'] * 3 * 3, np.float32, gpudata = op['vn_pointer']), \
			np.int32(op['vnum']), np.int32(op['fnum']), np.int32(3), \
			block = (1024, 1, 1))
		context.synchronize()

	def bs_coef_anim(self, outpath, coef, bs_lst, mesh_lst, tex_lst, use_tex = True, fps = 25):
		fourcc = cv.VideoWriter_fourcc(*'XVID')
		video_writer = cv.VideoWriter(outpath, fourcc, fps, (self.height, self.width))

		if use_tex:
			self.shader_tex.bind()
		else:
			self.shader_color.bind()

		op_lst = []
		for i in range(len(bs_lst)):
			op = self.upload_object(bs_lst[i], mesh_lst[i].faces, use_tex, mesh_lst[i].faces_tc, mesh_lst[i].tex_coords, tex_lst[i])
			op_lst.append(op)

		for i in range(len(coef)):
			glClearColor(1, 1, 1, 1)
			glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

			coef_gpu = self.move_to_gpu(coef[i].astype(np.float32))

			for j in range(len(mesh_lst)):
				self.interp_bs_object(op_lst[j], coef_gpu)
				glBindVertexArray(op_lst[j]['vao'])
				glDrawArrays(GL_TRIANGLES, 0, op_lst[j]['fnum'] * 3)
				glFlush()

			img_buffer = glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_UNSIGNED_BYTE)
			img = np.frombuffer(img_buffer, dtype = np.uint8).reshape(self.height, self.width, 3)[::-1]
			video_writer.write(img)

		video_writer.release()
