import numpy as np 
import scipy.sparse as sp 

from common.mesh import TriMesh 
from common.transform.pose_estimate import PoseEstimate

class DeformTransfer:
	def __init__(self):
		pass

	def cal_normal(self, verts, faces):
		v1 = verts[faces[:, 1]] - verts[faces[:, 0]]
		v2 = verts[faces[:, 2]] - verts[faces[:, 0]]
		normal = np.cross(v1, v2, axis = 1)
		normal /= np.sqrt(np.sum(normal ** 2, axis = 1))[:, np.newaxis] + 1e-12
		return normal.astype(np.float32)

	def cal_to_V_mat(self, faces, vn):
		fn = len(faces)
		row = np.column_stack((np.arange(fn) * 3, np.arange(fn) * 3, np.arange(fn) * 3 + 1, np.arange(fn) * 3 + 1, np.arange(fn) * 3 + 2)).flatten()
		col = np.column_stack((faces[:, 0], faces[:, 1], faces[:, 0], faces[:, 2], np.arange(fn) + vn)).flatten()
		data = np.ones((fn, 5), dtype = np.float32)
		data[:, [0, 2]] *= -1
		M = sp.coo_matrix((data.flatten(), (row, col)), shape = (3 * fn, fn + vn))
		return M

	def solve_mat_3_lst(self, mat_lst):
		a, b, c = mat_lst[:, :, 0], mat_lst[:, :, 1], mat_lst[:, :, 2]

		det = a[:, 0] * (b[:, 1] * c[:, 2] - c[:, 1] * b[:, 2]) - \
			a[:, 1] * (b[:, 0] * c[:, 2] - c[:, 0] * b[:, 2]) + \
			a[:, 2] * (b[:, 0] * c[:, 1] - c[:, 0] * b[:, 1])

		inv = np.zeros(mat_lst.shape, dtype = np.float32)
		inv[:, 0, 0] = (b[:, 1] * c[:, 2] - c[:, 1] * b[:, 2]) / det 
		inv[:, 0, 1] = (c[:, 0] * b[:, 2] - b[:, 0] * c[:, 2]) / det 
		inv[:, 0, 2] = (b[:, 0] * c[:, 1] - c[:, 0] * b[:, 1]) / det 

		inv[:, 1, 0] = (c[:, 1] * a[:, 2] - a[:, 1] * c[:, 2]) / det 
		inv[:, 1, 1] = (a[:, 0] * c[:, 2] - c[:, 0] * a[:, 2]) / det 
		inv[:, 1, 2] = (a[:, 1] * c[:, 0] - a[:, 0] * c[:, 1]) / det 

		inv[:, 2, 0] = (a[:, 1] * b[:, 2] - b[:, 1] * a[:, 2]) / det 
		inv[:, 2, 1] = (b[:, 0] * a[:, 2] - a[:, 0] * b[:, 2]) / det 
		inv[:, 2, 2] = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / det 

		return inv

	def transfer(self, source, source_def, target, faces, con_ind, con_verts, con_weight = 50):
		vn = len(source)
		fn = len(faces)
		trans_V = self.cal_to_V_mat(faces, len(source))

		s_n = self.cal_normal(source, faces)
		s_V_1 =  self.solve_mat_3_lst(trans_V.dot(np.vstack((source, s_n))).reshape((fn, 3, 3)))
		sd_n = self.cal_normal(source_def, faces)
		sd_V = trans_V.dot(np.vstack((source_def, sd_n))).reshape((fn, 3, 3))
		c = np.vstack((np.matmul(s_V_1, sd_V).reshape((fn * 3, 3)), con_verts * con_weight))

		t_n = self.cal_normal(target, faces)
		t_V_1 =  self.solve_mat_3_lst(trans_V.dot(np.vstack((target, t_n))).reshape((fn, 3, 3)))
		row = np.arange(fn * 3).repeat(3)
		col = np.tile(np.arange(fn * 3).reshape((-1, 3)), 3).flatten()
		A_1 = sp.coo_matrix((t_V_1.flatten(), (row, col)), shape = (fn * 3, fn * 3))
		A = A_1.dot(trans_V)
		
		A_con = sp.coo_matrix((np.full(len(con_ind), con_weight, dtype = np.float32), (np.arange(len(con_ind)), con_ind)), \
			shape = (len(con_ind), vn + fn))
		A = sp.vstack((A, A_con))

		x = sp.linalg.spsolve(A.T.dot(A), A.T.dot(c))
		return x[:vn]

	def transfer_bs_pose_est(self, source_lst, target, faces, con_num = 500):
		if not hasattr(self, 'pose_est'):
			self.pose_est = PoseEstimate()

		vn = len(target)
		fn = len(faces)
		trans_V = self.cal_to_V_mat(faces, len(target))

		t_n = self.cal_normal(target, faces)
		t_V_1 =  self.solve_mat_3_lst(trans_V.dot(np.vstack((target, t_n))).reshape((fn, 3, 3)))
		
		row = np.arange(fn * 3).repeat(3)
		col = np.tile(np.arange(fn * 3).reshape((-1, 3)), 3).flatten()
		A_1 = sp.coo_matrix((t_V_1.flatten(), (row, col)), shape = (fn * 3, fn * 3))
		A = A_1.dot(trans_V)
		A_fac = sp.linalg.splu(A.T.dot(A))

		s_n = self.cal_normal(source_lst[0], faces)
		s_V_1 =  self.solve_mat_3_lst(trans_V.dot(np.vstack((source_lst[0], s_n))).reshape((fn, 3, 3)))

		t_lst = []
		for i in range(1, len(source_lst)):
			sd_n = self.cal_normal(source_lst[i], faces)
			sd_V = trans_V.dot(np.vstack((source_lst[i], sd_n))).reshape((fn, 3, 3))

			c = np.matmul(s_V_1, sd_V).reshape((fn * 3, 3))
			x = A_fac.solve(A.T.dot(c))[:vn]

			con_ind = np.argsort(np.sqrt(np.sum((source_lst[0] - source_lst[i]) ** 2, axis = 1)))[:con_num]
			r, t = self.pose_est.svd_rt(x[con_ind], target[con_ind])
			x = x.dot(r.T) + t 
			t_lst.append(x)
			
		return np.array(t_lst)

	def transfer_bs_fix(self, source_lst, target, faces, con_ind, con_weight = 1):
		vn = len(target)
		fn = len(faces)
		trans_V = self.cal_to_V_mat(faces, len(target))

		t_n = self.cal_normal(target, faces)
		t_V_1 =  self.solve_mat_3_lst(trans_V.dot(np.vstack((target, t_n))).reshape((fn, 3, 3)))
		
		row = np.arange(fn * 3).repeat(3)
		col = np.tile(np.arange(fn * 3).reshape((-1, 3)), 3).flatten()
		A_1 = sp.coo_matrix((t_V_1.flatten(), (row, col)), shape = (fn * 3, fn * 3))
		A = A_1.dot(trans_V)

		I_lambda = sp.coo_matrix((np.ones(len(con_ind), dtype = np.float32) * con_weight, (np.arange(len(con_ind)), con_ind)), \
			shape = (len(con_ind), vn + fn))
		A = sp.vstack((A, I_lambda))
		A_fac = sp.linalg.splu(A.T.dot(A))

		s_n = self.cal_normal(source_lst[0], faces)
		s_V_1 =  self.solve_mat_3_lst(trans_V.dot(np.vstack((source_lst[0], s_n))).reshape((fn, 3, 3)))

		t_lst = []
		for i in range(1, len(source_lst)):
			sd_n = self.cal_normal(source_lst[i], faces)
			sd_V = trans_V.dot(np.vstack((source_lst[i], sd_n))).reshape((fn, 3, 3))

			c = np.vstack((np.matmul(s_V_1, sd_V).reshape((fn * 3, 3)), target[con_ind] * con_weight))
			x = A_fac.solve(A.T.dot(c))[:vn]

			t_lst.append(x)
			
		return t_lst


