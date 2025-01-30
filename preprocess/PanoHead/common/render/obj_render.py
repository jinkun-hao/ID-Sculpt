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


class GL_ObjRenderer:
    def __init__(self, width = 1000, height = 1000):
        self.width = width 
        self.height = height 

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

    def update_obj(self, mesh, texture):
        if mesh.vert_normal is None:
            mesh.cal_vert_normal()

        use_tex = True
        use_faces_tc = True

        if texture is None or mesh.tex_coords is None:
            use_tex = False 
            use_faces_tc = False
            shader = self.shader_color
        else:
            shader = self.shader_tex

        if mesh.faces_tc is None:
            use_faces_tc = False

        shader.bind()

        vao_vert = glGenVertexArrays(1)
        glBindVertexArray(vao_vert)

        vbo_pos = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
        if not use_faces_tc:
            glBufferData(GL_ARRAY_BUFFER, mesh.vert_num() * 4 * 3, mesh.vertices.astype(np.float32), GL_STATIC_DRAW)
        else:
            glBufferData(GL_ARRAY_BUFFER, mesh.face_num() * 3 * 3 * 4, mesh.vertices[mesh.faces].astype(np.float32), GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(0)

        vbo_normal = glGenBuffers(1) 
        glBindBuffer(GL_ARRAY_BUFFER, vbo_normal)
        if not use_faces_tc:
            glBufferData(GL_ARRAY_BUFFER, mesh.vert_num() * 4 * 3, mesh.vert_normal.astype(np.float32), GL_STATIC_DRAW)
        else:
            glBufferData(GL_ARRAY_BUFFER, mesh.face_num() * 3 * 3 * 4, mesh.vert_normal[mesh.faces].astype(np.float32), GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
        glEnableVertexAttribArray(1)

        if use_tex:
            vbo_tc = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_tc)
            if not use_faces_tc:
                glBufferData(GL_ARRAY_BUFFER, mesh.vert_num() * 4 * 2, mesh.tex_coords.astype(np.float32), GL_STATIC_DRAW)
            else:
                glBufferData(GL_ARRAY_BUFFER, mesh.face_num() * 3 * 2 * 4, mesh.tex_coords[mesh.faces_tc].astype(np.float32), GL_STATIC_DRAW)

            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
            glEnableVertexAttribArray(2)

            obj_tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, obj_tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture.shape[1], texture.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, texture)

        else:
            if mesh.vert_color is None:
                vert_color = np.full((mesh.vert_num(), 3), 0.83).astype(np.float32)
            else:
                vert_color = mesh.vert_color.astype(np.float32)

            vbo_color = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo_color)
            if not use_faces_tc:
                glBufferData(GL_ARRAY_BUFFER, mesh.vert_num() * 4 * 3, vert_color.astype(np.float32), GL_STATIC_DRAW)
            else:
                glBufferData(GL_ARRAY_BUFFER, mesh.vert_num() * 4 * 3 * 3, vert_color[mesh.faces].astype(np.float32), GL_STATIC_DRAW)

            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
            glEnableVertexAttribArray(2)

        if not use_faces_tc:
            ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.face_num() * 4 * 3, mesh.faces.astype(np.uint32), GL_STATIC_DRAW)

        shader.unbind()
        return vao_vert, use_tex, use_faces_tc

    def render_mesh_list(self, mesh_list, trans_mat, cam_mat, tex_list = None):
        self.shader_color.gl_transform(trans_mat, cam_mat)
        self.shader_tex.gl_transform(trans_mat, cam_mat)

        glClearColor(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for i in range(len(mesh_list)):
            if tex_list is None or tex_list[i] is None:
                vao, use_tex, use_faces_tc = self.update_obj(mesh_list[i], None)
            else:
                vao, use_tex, use_faces_tc = self.update_obj(mesh_list[i], tex_list[i])
            glBindVertexArray(vao)

            if not use_tex:
                self.shader_color.bind()
            else:
                self.shader_tex.bind()

            if not use_faces_tc:
                glDrawElements(GL_TRIANGLES, mesh_list[i].face_num() * 3, GL_UNSIGNED_INT, c_void_p(0))
            else:
                glDrawArrays(GL_TRIANGLES, 0, mesh_list[i].face_num() * 3)
            glFlush()

        img_buffer = glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_UNSIGNED_BYTE)
        img = np.frombuffer(img_buffer, dtype = np.uint8).reshape(self.height, self.width, 3)[::-1]
        return img

    def render_mesh_list_multi_view(self, mesh_list, tex_list = None, angles = [-pi/6, 0, pi/6]): 
        imgs = []
        for j in range(len(mesh_list)):
            if mesh_list[j].vert_normal is None:
                mesh_list[j].cal_vert_normal()
                
        for i in range(len(angles)):
            glClearColor(1, 1, 1, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            rot_mat = rotate_mat_euler_angle(0, angles[i], 0)

            for j in range(len(mesh_list)):
                tmp_mesh = mesh_list[j].copy()
                tmp_mesh.vertices = tmp_mesh.vertices.dot(rot_mat.T)
                tmp_mesh.vert_normal = tmp_mesh.vert_normal.dot(rot_mat.T)

                if tex_list is None or tex_list[j] is None:
                    vao, use_tex, use_faces_tc = self.update_obj(tmp_mesh, None)
                else:
                    vao, use_tex, use_faces_tc = self.update_obj(tmp_mesh, tex_list[j])
                glBindVertexArray(vao)

                if not use_tex:
                    self.shader_color.bind()
                else:
                    self.shader_tex.bind()

                if not use_faces_tc:
                    glDrawElements(GL_TRIANGLES, tmp_mesh.face_num() * 3, GL_UNSIGNED_INT, c_void_p(0))
                else:
                    glDrawArrays(GL_TRIANGLES, 0, tmp_mesh.face_num() * 3)
                glFlush()

            img_buffer = glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_UNSIGNED_BYTE)
            img = np.frombuffer(img_buffer, dtype = np.uint8).reshape(self.height, self.width, 3)[::-1]
            imgs.append(img)

        return imgs