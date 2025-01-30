import numpy as np 

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import *
from OpenGL.GLU import *
import glfw

from glm import perspective, atan, lookAt, vec3, radians
from math import pi 
from common.transform.transform import rotate_mat_euler_angle


vert_shader_tex = """
#version 450 core
layout (location = 0) in vec3 vert_pos;
layout (location = 1) in vec3 vert_normal;
layout (location = 2) in vec2 tex_coords;

out vec3 Normal;
out vec3 FragPos;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	gl_Position = projection * view * model * vec4(vert_pos, 1.0f);
	FragPos = vec3(model * vec4(vert_pos, 1.0f));
	Normal = normalize(vec3(model * vec4(vert_normal, 0.0f)));  
	TexCoord = tex_coords;
}
"""

frag_shader_tex = """
#version 450 core
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform vec4 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;

uniform sampler2D obj_texture;

out vec4 color;

void main(){
	float ambient_S = 0.3f;
	vec4 ambient = ambient_S * lightColor;

	vec3 norm = normalize(Normal);
	vec3 light_dir = normalize(lightPos - FragPos);

	float diffuse_S = 0.55f;
	float diff = max(dot(norm, light_dir), 0.0);
	vec4 diffuse = diffuse_S * diff * lightColor;

	float specular_S = 0.04f;
	vec3 view_dir = normalize(viewPos - FragPos);
	vec3 reflect_dir = normalize(reflect(-light_dir, norm));

	float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
	vec4 specular = specular_S * spec * lightColor;

	vec4 vert_color = texture(obj_texture, TexCoord);

	color = (ambient + diffuse + specular) * vert_color;
}
"""

frag_shader_color = """
#version 450 core
in vec3 FragPos;
in vec3 Normal;
in vec3 Color;

uniform vec4 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;

out vec4 color;

void main(){
	float ambient_S=0.3f;
	vec4 ambient=ambient_S*lightColor;

	vec3 norm=normalize(Normal);
	vec3 light_dir=normalize(lightPos-FragPos);

	float diffuse_S=0.55f;
	float diff=max(dot(norm,light_dir),0.0);
	vec4 diffuse=diffuse_S*diff*lightColor;

	float specular_S=0.04f;
	vec3 view_dir=normalize(viewPos-FragPos);
	vec3 reflect_dir=normalize(reflect(-light_dir,norm));

	float spec=pow(max(dot(view_dir,reflect_dir),0.0),32);
	vec4 specular=specular_S*spec*lightColor;

	color=(ambient+diffuse+specular)* vec4(Color, 1.0);
}
"""

vert_shader_color = """
#version 450 core
layout (location = 0) in vec3 vert_pos;
layout (location = 1) in vec3 vert_normal;
layout (location = 2) in vec3 vert_color;

out vec3 Normal;
out vec3 FragPos;
out vec3 Color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	gl_Position = projection * view * model * vec4(vert_pos, 1.0f);
	FragPos = vec3(model * vec4(vert_pos, 1.0f));
	Normal = normalize(vec3(model * vec4(vert_normal, 0.0f)));  
	Color = vert_color;
}
"""

vert_shader_wire = """
#version 450 core 
layout (location = 0) in vec3 vert_pos;
layout (location = 1) in vec3 vert_normal;
layout (location = 2) in vec3 vert_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vs_out_struct{
	vec3 Normal;
	vec3 Color;
	vec3 Pos;
} vs_out;

void main(){
	gl_Position = projection * view * model * vec4(vert_pos, 1.0f);
	vs_out.Normal = normalize(vec3(model * vec4(vert_normal, 0.0f)));
	vs_out.Pos = vec3(model * vec4(vert_pos, 1.0f));
	vs_out.Color = vert_color;
}
"""

geom_shader_wire = """
#version 450 core 
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

uniform vec2 WIN_SCALE;

in vs_out_struct{
	vec3 Normal;
	vec3 Color;
	vec3 Pos;
} gs_in[];

out vec3 dist;
out vec3 pos;
out vec3 color;
out vec3 normal;

void main(){
	vec4 p0_3d = gl_in[0].gl_Position;
	vec4 p1_3d = gl_in[1].gl_Position;
	vec4 p2_3d = gl_in[2].gl_Position;

	vec2 p0 = WIN_SCALE *  p0_3d.xy / p0_3d.w;
	vec2 p1 = WIN_SCALE * p1_3d.xy / p1_3d.w;
	vec2 p2 = WIN_SCALE * p2_3d.xy / p2_3d.w;

	vec2 v0 = p2 - p1;
	vec2 v1 = p2 - p0;
	vec2 v2 = p1 - p0;

	float area = abs(v1.x * v2.y - v1.y * v2.x);

	dist = vec3(area / length(v0), 0.0f, 0.0f);
	pos = gs_in[0].Pos;
	color = gs_in[0].Color;
	normal = gs_in[0].Normal;
	gl_Position = p0_3d;
	EmitVertex();

	dist = vec3(0, area / length(v1), 0);
	pos = gs_in[1].Pos;
	color = gs_in[1].Color;
	normal = gs_in[1].Normal;
	gl_Position = p1_3d;
	EmitVertex();

	dist = vec3(0, 0, area / length(v2));
	pos = gs_in[2].Pos;
	color = gs_in[2].Color;
	normal = gs_in[2].Normal;
	gl_Position = p2_3d;
	EmitVertex();

	EndPrimitive();
}

"""

frag_shader_wire = """
#version 450 core
in vec3 dist;
in vec3 pos;
in vec3 normal;
in vec3 color;

uniform vec4 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec4 lineColor;

out vec4 frag_color;

void main(){
	float nearD = min(min(dist[0], dist[1]), dist[2]);
	float edgeIntensity = exp2(-1.0f * nearD * nearD);

	float ambient_S = 0.3f;
	vec4 ambient = ambient_S * lightColor;

	vec3 light_dir = normalize(lightPos - pos);
	//vec3 light_dir = normalize(vec3(0.0f, 0.0f, -10.0f));

	float diffuse_S = 0.55f;
	float diff = max(dot(normal, light_dir), 0.0f);
	vec4 diffuse = diffuse_S * diff * lightColor;

	float specular_S = 0.04f;
	vec3 view_dir = normalize(viewPos - pos);
	vec3 reflect_dir = normalize(reflect(-light_dir, normal));

	float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
	vec4 specular = specular_S * spec * lightColor;

	vec4 ori_color = (ambient + diffuse + specular) * vec4(color, 1.0);
	frag_color = edgeIntensity * lineColor + (1.0 - edgeIntensity) * ori_color;
}
"""

vert_shader_bgi = """
#version 450 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 tex_coord;

out vec2 tex_coords;

void main(){
	gl_Position = vec4(pos, 1.0);
	tex_coords = tex_coord;
}

"""

frag_shader_bgi = """
#version 450 core
in vec2 tex_coords;
out vec4 color;

uniform sampler2D texture1;

void main(){
	color = texture(texture1, tex_coords);
}
"""

class Shader:
	def __init__(self, shader_type = 'vert_color', width = 1000, height = 1000):
		if not shader_type in ['vert_color', 'texture', 'wireframe', 'background']:
			print('shader type error')
			return
		self.shader_type = shader_type
		self.width = width
		self.height = height

		if shader_type == 'vert_color':
			shader_vert = compileShader(vert_shader_color, GL_VERTEX_SHADER)
			shader_frag = compileShader(frag_shader_color, GL_FRAGMENT_SHADER)
		elif shader_type == 'texture':
			shader_vert = compileShader(vert_shader_tex, GL_VERTEX_SHADER)
			shader_frag = compileShader(frag_shader_tex, GL_FRAGMENT_SHADER)
		elif shader_type == 'wireframe':
			shader_vert = compileShader(vert_shader_wire, GL_VERTEX_SHADER)
			shader_geom = compileShader(geom_shader_wire, GL_GEOMETRY_SHADER)
			shader_frag = compileShader(frag_shader_wire, GL_FRAGMENT_SHADER)
		elif shader_type == 'background':
			shader_vert = compileShader(vert_shader_bgi, GL_VERTEX_SHADER)
			shader_frag = compileShader(frag_shader_bgi, GL_FRAGMENT_SHADER)

		if shader_type in ['vert_color', 'texture', 'background']:
			self.program_id = compileProgram(shader_vert, shader_frag)
		else:
			self.program_id = compileProgram(shader_vert, shader_frag, shader_geom)

		self.bind()
		if self.shader_type in ['vert_color', 'texture', 'wireframe']:
			self.gl_transform_default()
			self.gl_light_default()

		if self.shader_type == 'wireframe':
			self.set_float4("lineColor", np.array([0.15, 0.15, 0.15, 1], dtype = np.float32))
			self.set_float2("WIN_SCALE", np.array([self.width, self.height], dtype = np.float32))
		
		self.unbind()

	def set_mat4(self, attr_name, attr, transpose = False):
		loc = glGetUniformLocation(self.program_id, attr_name)
		glUniformMatrix4fv(loc, 1, transpose, np.asarray(attr, dtype = np.float32))
	
	def set_float3(self, attr_name, attr):
		attr = attr.astype(np.float32)
		loc = glGetUniformLocation(self.program_id, attr_name)
		glUniform3f(loc, attr[0], attr[1], attr[2])
	
	def set_float4(self, attr_name, attr):
		attr = attr.astype(np.float32)
		loc = glGetUniformLocation(self.program_id, attr_name)
		glUniform4f(loc, attr[0], attr[1], attr[2], attr[3])

	def set_float2(self, attr_name, attr):
		attr = attr.astype(np.float32)
		loc = glGetUniformLocation(self.program_id, attr_name)
		glUniform2f(loc, attr[0], attr[1])

	def bind(self):
		glUseProgram(self.program_id)

	def unbind(self):
		glUseProgram(0)

	def gl_transform(self, trans_mat, cam_mat):
		if not self.shader_type in ['vert_color', 'texture', 'wireframe']:
			return
		"""
		camera intrinsic matrix should be like
		[
		f 0 cx
		0 f cy
		0 0 1
		]
		"""
		projection = perspective(2 * atan(self.height / 2 / cam_mat[0, 0]), float(self.width) / self.height, 0.1, 1000)
		projection = np.asarray(projection).T 
		projection[0, 2] = 1 - 2 * cam_mat[0, 2] / self.width
		projection[1, 2] = -1 + 2 * cam_mat[1, 2] / self.height 

		self.bind()
		self.set_mat4("projection", projection, transpose = True)
		self.set_mat4("model", trans_mat, transpose = True)
		self.unbind()

	def gl_transform_default(self):
		trans_mat = np.eye(4)
		rot_mat = rotate_mat_euler_angle(pi, 0, 0)
		trans_mat[:3, :3] = rot_mat 
		trans_mat[2, 3] = 5.5
		# trans_mat[1, 3] = -0.5
 
		projection = perspective(radians(30.0), 1, 0.1, 1000)
		view = lookAt(vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, -1, 0))

		self.set_mat4("model", trans_mat, transpose = True)
		self.set_mat4("projection", projection, transpose = False)
		self.set_mat4("view", view)
		glViewport(0, 0, self.width, self.height)

	def gl_light_default(self):
		self.set_float3("lightPos", np.array([0, 0, -10], dtype = np.float32))
		self.set_float3("viewPos", np.array([0, 0, 0], dtype = np.float32))
		self.set_float4("lightColor", np.array([1, 1, 1, 1], dtype = np.float32))