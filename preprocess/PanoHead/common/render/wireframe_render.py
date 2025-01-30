from OpenGL.GL import * 
from OpenGL.GLUT import * 
from OpenGL.GL.shaders import * 
import glfw 

import cv2 as cv 
import numpy as np 
from math import pi 

from common.mesh import TriMesh 
from common.transform.transform import rotate_mat_euler_angle 
from common.render.shader import Shader

class GL_WireFrameRenderer:
	def __init__(self, width = 1000, height = 1000):
		self.width = width 
		self.height = height 

		if not glfw.init():
			print('glfw init fail')
			return 

		glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
		glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
		glfw.window_hint(glfw.SAMPLES, 8)
		glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

		self.window = glfw.create_window(self.width, self.height, "opengl", None, None)

		if not self.window:
			glfw.terminate()
			print('glfw init window fail')
			return 

		glfw.make_context_current(self.window)

		self.shader = Shader('wireframe', width, height)
		self.shader.bind()
	
		glEnable(GL_MULTISAMPLE)
		glEnable(GL_DEPTH_TEST)

	def update_obj(self, mesh):
		vao_vert = glGenVertexArrays(1)
		glBindVertexArray(vao_vert)

		vbo_pos = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
		glBufferData(GL_ARRAY_BUFFER, mesh.vert_num() * 4 * 3, mesh.vertices.astype(np.float32), GL_STATIC_DRAW)
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
		glEnableVertexAttribArray(0)

		vbo_normal = glGenBuffers(1)
		if mesh.vert_normal is None:
			mesh.cal_vert_normal()
		glBindBuffer(GL_ARRAY_BUFFER, vbo_normal)
		glBufferData(GL_ARRAY_BUFFER, mesh.vert_num() * 4 * 3, mesh.vert_normal.astype(np.float32), GL_STATIC_DRAW)
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
		glEnableVertexAttribArray(1)

		if mesh.vert_color is None:
			vert_color = np.full((mesh.vert_num(), 3), 0.83).astype(np.float32)
		else:
			vert_color = mesh.vert_color.astype(np.float32)
		vbo_color = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, vbo_color)
		glBufferData(GL_ARRAY_BUFFER, mesh.vert_num() * 4 * 3, vert_color.astype(np.float32), GL_STATIC_DRAW)
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
		glEnableVertexAttribArray(2)

		ebo = glGenBuffers(1)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.face_num() * 4 * 3, mesh.faces.astype(np.uint32), GL_STATIC_DRAW)

		return vao_vert

	def render(self, mesh, cam_mat, trans_mat):

		self.shader.gl_transform(trans_mat, cam_mat)
		self.shader.bind()
		
		glClearColor(1, 1, 1, 1)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		vao = self.update_obj(mesh)

		glBindVertexArray(vao)
		glDrawElements(GL_TRIANGLES, mesh.face_num() * 3, GL_UNSIGNED_INT, c_void_p(0))
		glFlush()

		img_buffer = glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_UNSIGNED_BYTE)
		img = np.frombuffer(img_buffer, dtype = np.uint8).reshape(self.height, self.width, 3)[::-1]

		return img

	def render_mesh_multi_view(self, mesh, angles = [-pi/6, 0, pi/6]):
		mesh_copy = TriMesh(mesh.vertices.copy(), mesh.faces.copy())
		if mesh.vert_normal is None:
			mesh.cal_vert_normal()
		mesh_copy.vert_normal = mesh.vert_normal.copy()

		self.shader.bind()

		imgs = []
		for i in range(len(angles)):
			glClearColor(1, 1, 1, 1)
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

			rot_mat = rotate_mat_euler_angle(0, angles[i], 0)
			mesh_copy.vertices = mesh_copy.vertices.dot(rot_mat.T)
			mesh_copy.vert_normal = mesh_copy.vert_normal.dot(rot_mat.T)

			vao = self.update_obj(mesh_copy)
			glBindVertexArray(vao)
			glDrawElements(GL_TRIANGLES, mesh_copy.face_num() * 3, GL_UNSIGNED_INT, c_void_p(0))
			glFlush()

			img_buffer = glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_UNSIGNED_BYTE)
			img = np.frombuffer(img_buffer, dtype = np.uint8).reshape(self.height, self.width, 3)[::-1]
			imgs.append(img)

			mesh_copy.vertices = mesh.vertices.copy()
			mesh_copy.vert_normal = mesh.vert_normal.copy()

		return imgs