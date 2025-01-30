import numpy as np 

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import *
from OpenGL.GLU import *
import glfw

from glm import perspective, atan, lookAt, vec3, radians
from math import pi 
from common.transform.transform import rotate_mat_euler_angle

vert_shader_normal = """
#version 450 core

layout (location = 0) in vec3 vert_pos;
layout (location = 1) in vec3 vert_normal;
layout (location = 2) in vec2 tex_coords;
layout (location = 3) in vec3 tangent;

out vec3 FragPos;
out vec2 TexCoords;
out vec3 tangentLightPos;
out vec3 tangentViewPos;
out vec3 tangentFragPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec3 lightPos;
uniform vec3 viewPos;

void main(){
	FragPos = vec3(model * vec4(vert_pos, 1.0));
	TexCoords = tex_coords;
	gl_Position = projection * view * model * vec4(vert_pos, 1.0f);

	mat3 normalMatrix = transpose(inverse(mat3(model)));
	vec3 T = normalize(normalMatrix * tangent);
	vec3 N = normalize(normalMatrix * vert_normal);
	T = normalize(T - dot(T, N) * N);
	vec3 B = cross(N, T);

	mat3 TBN = transpose(mat3(T, B, N));
	tangentLightPos = TBN * lightPos;
	tangentViewPos = TBN * viewPos;
	tangentFragPos = TBN * FragPos;
}

"""

frag_shader_normal = """
#version 450 core

out vec4 FragColor;

in vec3 FragPos;
in vec2 TexCoords;
in vec3 tangentLightPos;
in vec3 tangentViewPos;
in vec3 tangentFragPos;

uniform sampler2D textureMap;
uniform sampler2D normalMap;

void main(){
	vec3 color = texture(textureMap, TexCoords).rgb;
	vec3 ambient = 0.1 * color;
	
	vec3 normal = texture(normalMap, TexCoords).rgb;
	normal = normalize(normal * 2.0 - 1.0);

	vec3 lightDir = normalize(tangentLightPos - tangentFragPos);
	float diff = max(dot(lightDir, normal), 0.0);
	vec3 diffuse = diff * color;

	vec3 viewDir = normalize(tangentViewPos - tangentFragPos);
	vec3 reflectDir = reflect(-lightDir, normal);
	vec3 halfwayDir = normalize(lightDir + viewDir);
	float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
	vec3 specular = vec3(0.1) * spec;

	FragColor = vec4(ambient + diffuse + specular, 1.0);
}

"""

class NormalMapRenderer:
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

		shader_vert = compileShader(vert_shader_normal, GL_VERTEX_SHADER)
		shader_frag = compileShader(frag_shader_normal, GL_FRAGMENT_SHADER)

		self.program_id = compileProgram(shader_vert, shader_frag)
		glUseProgram(self.program_id)

		self.gl_transform_default()

		self.set_float3("lightPos", np.array([0, 0, -10], dtype = np.float32))
		self.set_float3("viewPos", np.array([0, 0, 0], dtype = np.float32))

		glUniform1i(glGetUniformLocation(self.program_id, "textureMap"), 0)
		glUniform1i(glGetUniformLocation(self.program_id, "normalMap"), 1)

		glEnable(GL_MULTISAMPLE)
		glEnable(GL_DEPTH_TEST)

	def set_mat4(self, attr_name, attr, transpose = False):
		loc = glGetUniformLocation(self.program_id, attr_name)
		glUniformMatrix4fv(loc, 1, transpose, np.asarray(attr, dtype = np.float32))

	def set_float3(self, attr_name, attr):
		attr = attr.astype(np.float32)
		loc = glGetUniformLocation(self.program_id, attr_name)
		glUniform3f(loc, attr[0], attr[1], attr[2])

	def gl_transform_default(self):
		trans_mat = np.eye(4)
		rot_mat = rotate_mat_euler_angle(pi, 0, 0)
		trans_mat[:3, :3] = rot_mat 
		trans_mat[2, 3] = 3

		projection = perspective(radians(30.0), 1, 0.1, 1000)
		view = lookAt(vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, -1, 0))

		self.set_mat4("model", trans_mat, transpose = True)
		self.set_mat4("projection", projection, transpose = False)
		self.set_mat4("view", view)
		glViewport(0, 0, self.width, self.height)

	def cal_vert_tangent(self, mesh):
		edge1 = mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]]
		edge2 = mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 0]]

		deltaUV1 = mesh.tex_coords[mesh.faces_tc[:, 1]] - mesh.tex_coords[mesh.faces_tc[:, 0]]
		deltaUV2 = mesh.tex_coords[mesh.faces_tc[:, 2]] - mesh.tex_coords[mesh.faces_tc[:, 0]]

		f = 1.0 / (deltaUV1[:, 0] * deltaUV2[:, 1] - deltaUV1[:, 1] * deltaUV2[:, 0])
		tangent_x = f * deltaUV2[:, 1] * edge1[:, 0] - deltaUV1[:, 1] * edge2[:, 0]
		tangent_y = f * deltaUV2[:, 1] * edge1[:, 1] - deltaUV1[:, 1] * edge2[:, 1]
		tangent_z = f * deltaUV2[:, 1] * edge1[:, 2] - deltaUV1[:, 1] * edge2[:, 2]

		tangent = np.column_stack((tangent_x, tangent_y, tangent_z))

		vert_tan = np.zeros(mesh.vertices.shape, dtype = np.float32)
		for i in range(len(mesh.faces)):
			vert_tan[mesh.faces[i]] += tangent[i]

		vert_tan /= np.sqrt(np.sum(vert_tan ** 2, axis = 1))[:, np.newaxis] + 1e-12

		return vert_tan

	def upload_obj(self, mesh, texture, normal_map):
		if mesh.vert_normal is None:
			mesh.cal_vert_normal()

		tangent = self.cal_vert_tangent(mesh)
		# tangent = np.transpose(tangent.repeat(3).reshape((-1, 3, 3)), (0, 2, 1))
		tangent = tangent[mesh.faces]

		glUseProgram(self.program_id)

		vao = glGenVertexArrays(1)
		glBindVertexArray(vao)

		vbo_pos = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
		glBufferData(GL_ARRAY_BUFFER, mesh.face_num() * 3 * 3 * 4, mesh.vertices[mesh.faces].astype(np.float32), GL_STATIC_DRAW)
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
		glEnableVertexAttribArray(0)

		vbo_normal = glGenBuffers(1) 
		glBindBuffer(GL_ARRAY_BUFFER, vbo_normal)
		glBufferData(GL_ARRAY_BUFFER, mesh.face_num() * 3 * 3 * 4, mesh.vert_normal[mesh.faces].astype(np.float32), GL_STATIC_DRAW)
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
		glEnableVertexAttribArray(1)

		vbo_tc = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, vbo_tc)
		glBufferData(GL_ARRAY_BUFFER, mesh.face_num() * 3 * 2 * 4, mesh.tex_coords[mesh.faces_tc].astype(np.float32), GL_STATIC_DRAW)
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
		glEnableVertexAttribArray(2)

		vbo_tan = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, vbo_tan)
		glBufferData(GL_ARRAY_BUFFER, mesh.face_num() * 3 * 3 * 4, tangent.astype(np.float32), GL_STATIC_DRAW)
		glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
		glEnableVertexAttribArray(3)

		tex_id = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, tex_id)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture.shape[1], texture.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, texture)
		glGenerateMipmap(GL_TEXTURE_2D)

		normal_map_id = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, normal_map_id)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, normal_map.shape[1], normal_map.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, normal_map)
		glGenerateMipmap(GL_TEXTURE_2D)

		return vao, tex_id, normal_map_id

	def render_mesh(self, mesh, texture, normal_map):
		glClearColor(1, 1, 1, 1)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		vao, tex_id, normal_map_id = self.upload_obj(mesh, texture, normal_map)

		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, tex_id)
		glActiveTexture(GL_TEXTURE1)
		glBindTexture(GL_TEXTURE_2D, normal_map_id)

		glBindVertexArray(vao)
		glDrawArrays(GL_TRIANGLES, 0, mesh.face_num() * 3)
		glFlush()

		img_buffer = glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_UNSIGNED_BYTE)
		img = np.frombuffer(img_buffer, dtype = np.uint8).reshape(self.height, self.width, 3)[::-1]

		return img


