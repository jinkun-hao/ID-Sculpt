import numpy as np 
from math import sin, cos, pi 
from scipy.optimize import leastsq

from common.mesh import TriMesh 
from common.cor.correspondences import Correspondences
from common.cor.octree import OCTree
from common.transform.transform import rotate_mat_euler_angle

class PoseEstimate:
	def __init__(self, cor_type = 'points'):
		self.cor = Correspondences() 
		self.cor_type = cor_type

	def ransac_svd(self, src, tgt, ind_ini = None, max_iters = 30, dist_thres = 0.05):
		def cal_inline_verts(r, s, t):
			src_trans = src.dot(r.T) * s + t 
			dist = np.sqrt(np.sum((src_trans - tgt) ** 2, axis = 1))
			return np.where(dist < dist_thres)[0]

		if ind_ini is None:
			ind = np.arange(len(src))
			np.random.shuffle(ind)
			ind_ini = ind[:4]

		r_opt, s_opt, t_opt = self.svd(src[ind_ini], tgt[ind_ini])
		inline_ind = cal_inline_verts(r_opt, s_opt, t_opt)

		ids = np.arange(len(src))
		for i in range(max_iters):
			np.random.shuffle(ids)
			r, s, t = self.svd(src[ids[:4]], tgt[ids[:4]])
			
			ind = cal_inline_verts(r, s, t)
			if len(ind) > len(inline_ind):
				r_opt, s_opt, t_opt = r, s, t 
				inline_ind = ind

		return self.svd(src[inline_ind], tgt[inline_ind])
		# return r_opt, s_opt, t_opt

	def svd(self, source, target):
		u_s = np.mean(source, axis = 0)
		u_t = np.mean(target, axis = 0)

		m_s = source - u_s 
		m_t = target - u_t 

		sigma_s = np.mean(np.sum(m_s ** 2, axis = 1))
		sigma_t = np.mean(np.sum(m_t ** 2, axis = 1))

		sigma_st = m_s.T.dot(m_t).T / len(source)

		U, D, Vt = np.linalg.svd(sigma_st)
		S = np.eye(3)
		if np.linalg.det(sigma_st) < 0:
			S[2, 2] = -1

		R = U.dot(S).dot(Vt)
		s = np.trace(np.diag(D).dot(S)) / sigma_s
		t = np.reshape(u_t, (-1, 1)) - s * R.dot(np.reshape(u_s, (-1, 1)))
		return R, s, t.flatten()

	def svd_rt(self, source, target):
		u_s = np.mean(source, axis = 0)
		u_t = np.mean(target, axis = 0)

		m_s = source - u_s 
		m_t = target - u_t 

		S = m_s.T.dot(m_t) 
		U, s, Vt = np.linalg.svd(S)

		det = np.linalg.det(Vt.T.dot(U.T))
		sigma = np.eye(3)
		sigma[2, 2] = det 
		r = Vt.T.dot(sigma).dot(U.T)

		t = u_t - r.dot(u_s[:, np.newaxis]).flatten()
		return r, t

	def pose_est(self, src, tgt, octree = None, max_iters = 20, dist_threshold = 0.01, normal_threshold = 0.5):
		def residual(pose, src_verts, src_normals, tgt_verts):
			rot = rotate_mat_euler_angle(pose[0], pose[1], pose[2])
			verts_trans = src_verts.dot(rot.T) * pose[6] + pose[3: 6]
			normals_trans = src_normals.dot(rot.T)
			return np.sum((verts_trans - tgt_verts) * normals_trans, axis = 1)

		if src.vert_normal is None:
			src.cal_vert_normal()

		if octree is None:
			octree = OCTree()
			octree.from_triangles(tgt.vertices, tgt.faces, np.arange(tgt.face_num()))

		src_verts = src.vertices.copy()
		src_normals = src.vert_normal.copy()
		pose = np.array([0, 0, 0, 0, 0, 0, 1]).astype(np.float32)

		for i in range(max_iters):
			print('pose estimate iters:', i, pose)
			rot = rotate_mat_euler_angle(pose[0], pose[1], pose[2])
			trans = src_verts.dot(rot.T) * pose[6] + pose[3: 6]
			normals = src_normals.dot(rot.T)
			normals /= (np.sqrt(np.sum(normals ** 2, axis = 1))[:, np.newaxis] + 1e-12) 

			tmp_mesh = TriMesh(vertices = trans, faces = src.faces)
			src_ind, tgt_face_ind, bary_coords = self.cor.nearest_tri_normal(tmp_mesh, tgt, octree, dist_threshold = dist_threshold, normal_threshold = normal_threshold)
			print(src_ind.shape)
			cor_verts = np.sum(tgt.vertices[tgt.faces[tgt_face_ind]] * bary_coords[:, :, np.newaxis], axis = 1)

			pose, flag = leastsq(residual, pose, args = (src_verts[src_ind], src_normals[src_ind], cor_verts))

		return pose

	def derivate_rigid_normal(self, src, src_normal, tgt, param, weights):
		jac = np.zeros((len(src), 7), dtype = np.float32)
		rmat = rotate_mat_euler_angle(param[0], param[1], param[2])
		jac[:, 3:6] = src_normal.dot(rmat.T)
		jac[:, 6] = np.sum(src * src_normal, axis = 1)

		sx, sy, sz = sin(param[0]), sin(param[1]), sin(param[2])
		cx, cy, cz = cos(param[0]), cos(param[1]), cos(param[2])

		jac_r1 = np.array([0, cx*sy + sx*cy*sz, -sx*sy + cx*cy*sz, \
			0, -sy*cz, -cz*cx, \
			0, cx*cy - sx*sy*sz, -sx*cy-cx*sy*sz]).reshape((3, 3))
		jac_r2 = np.array([-sy*cz, sx*cy+cx*sy*sz, cx*cy-sx*sy*sz, \
			0, 0, 0, \
			-cy*cz, -sx*sy+cx*cy*sz, -cx*sy-sx*cy*sz]).reshape((3, 3))
		jac_r3 = np.array([-cy*sz, -cx*cy*cz, cy*sx*cz, \
			cz, -cx*sz, sx*sz, \
			sz*sy, cx*sy*cz, -sx*sy*cz]).reshape((3, 3))
		jac[:, 0] = np.sum((param[3:6] - tgt).dot(jac_r1) * src_normal, axis = 1)
		jac[:, 1] = np.sum((param[3:6] - tgt).dot(jac_r2) * src_normal, axis = 1)
		jac[:, 2] = np.sum((param[3:6] - tgt).dot(jac_r3) * src_normal, axis = 1)

		jac = jac * weights

		res = np.sum((src.dot(rmat.T) * param[6] + param[3:6] - tgt) * src_normal.dot(rmat.T), axis = 1).reshape((-1, 1))
		res = res * weights
		
		return jac, res

	def pose_est_deriv(self, src, src_normal, tgt, param, weights, max_iters = 5):
		for i in range(max_iters):
			jac, res = self.derivate_rigid_normal(src, src_normal, tgt, param, weights)
			det = np.linalg.inv(jac.T.dot(jac)).dot(jac.T.dot(res)).flatten()
			param = param - det

			if np.max(np.abs(det)) < 1e-5:
				break

		return param