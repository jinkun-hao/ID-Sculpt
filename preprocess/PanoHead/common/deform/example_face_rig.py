import numpy as np 
import scipy.sparse as sp
import cv2 as cv

from common.mesh import TriMesh

vn = 4501
fn = 8684
nbs = 51

class ExampleFacialRig:
	def __init__(self):
		self.bs = np.load('../blendshape/bfm_topo_region.npz')['bs']

		region = TriMesh()
		region.load('../align/head_deformed_align_normal.obj')
		region_ind = np.loadtxt('../region/region_in_head.txt', dtype = np.int32)
		region.del_by_vert_ind(region_ind)
		self.faces = region.faces

		MA_0 = self.cal_M(self.bs[0], self.faces)
		MA_0_1 = np.linalg.inv(MA_0)
		self.GA_lst = []
		self.w_lst = []
		for i in range(1, len(self.bs)):
			MA_i = self.cal_M(self.bs[i], self.faces)
			MA_i_f = np.sqrt(np.sum((MA_i - MA_0)[:, :, :2] ** 2))
			self.GA_lst.append(np.matmul(MA_i, MA_0_1))
			self.w_lst.append(((1 + MA_i_f) / (0.1 + MA_i_f)) ** 2)
		self.GA_lst = np.array(self.GA_lst)
		self.w_lst = np.array(self.w_lst) 

	def cal_M(self, verts, faces):
		M = np.zeros((fn, 3, 3), dtype = np.float32)
		M[:, :, 0] = verts[faces[:, 1]] - verts[faces[:, 0]]
		M[:, :, 1] = verts[faces[:, 2]] - verts[faces[:, 0]]
		M[:, :, 2] = np.cross(M[:, :, 0], M[:, :, 1], axis = 1)
		M[:, :, 2] /= np.sqrt(np.sum(M[:, :, 2] ** 2, axis = 1))[:, np.newaxis] + 1e-12 
		return M 

	def est_bs_coef(self, b0, bias, s_lst, con_lst, gamma = 10):
		x_lst = []
		for i in range(len(s_lst)):
			A = np.vstack((bias.reshape((nbs, -1)).T, np.eye(nbs) * gamma))
			b = np.vstack(((s_lst[i] - b0).reshape((-1, 1)), con_lst[i][:, np.newaxis] * gamma))
			x = np.linalg.solve(A.T.dot(A), A.T.dot(b))
			x_lst.append(x.flatten())
		return np.array(x_lst)

	def cal_to_V_mat(self, faces, vn):
		row = np.arange(fn * 2).repeat(2)
		col = np.column_stack((faces[:, 0], faces[:, 1], faces[:, 0], faces[:, 2])).flatten()
		data = np.ones((fn, 4), dtype = np.float32)
		data[:, [0, 2]] *= -1 
		trans_V = sp.coo_matrix((data.flatten(), (row, col)), shape = (2 * fn, vn))
		return trans_V

	def recon_mesh(self, M, faces, vn, con_ind, con_verts, con_weight = 50):
		if not hasattr(self, 'trans_V'):
			self.trans_V = self.cal_to_V_mat(faces, vn)

		A_con = sp.coo_matrix((np.full(len(con_ind), con_weight, dtype = np.float32), (np.arange(len(con_ind)), con_ind)), \
			shape = (len(con_ind), vn))
		A = sp.vstack((self.trans_V, A_con))
		b = np.vstack((M.reshape((fn * 2, -1)), con_verts * con_weight))

		x = sp.linalg.spsolve(A.T.dot(A), A.T.dot(b))
		return x 

	def update_bs(self, b0, s_lst, coef_lst, faces, beta):
		ns = len(s_lst)
		MS_lst = []
		for i in range(ns):
			MS_i = np.zeros((fn, 3, 2), dtype = np.float32)
			MS_i[:, :, 0] = s_lst[i][faces[:, 1]] - s_lst[i][faces[:, 0]]
			MS_i[:, :, 1] = s_lst[i][faces[:, 2]] - s_lst[i][faces[:, 0]]
			MS_lst.append(MS_i)
		MS_lst = np.array(MS_lst)

		MB_0 = np.zeros((fn, 3, 2), dtype = np.float32)
		MB_0[:, :, 0] = b0[faces[:, 1]] - b0[faces[:, 0]]
		MB_0[:, :, 1] = b0[faces[:, 2]] - b0[faces[:, 0]]

		b_fit = MS_lst - MB_0
		b_reg = (np.matmul(self.GA_lst, MB_0[np.newaxis, :, :, :]) - MB_0) * self.w_lst[:, np.newaxis, np.newaxis, np.newaxis]

		A_fit = np.zeros((ns * 3, nbs * 3), dtype = np.float32)
		A_fit[np.tile(np.arange(3 * ns).reshape((-1, 3)), nbs).flatten(), np.tile(np.arange(3 * nbs), ns).flatten()] = coef_lst.flatten().repeat(3)
		A_reg = np.diag(self.w_lst.repeat(3)) * beta

		A = np.vstack((A_fit, A_reg))
		A_solve = np.linalg.inv(A.T.dot(A)).dot(A.T)
		
		M = np.zeros((nbs, fn, 3, 2), dtype = np.float32)
		for i in range(fn):
			b = np.vstack((b_fit[:, i, :, :].reshape((-1, 2)), beta * b_reg[:, i, :, :].reshape((-1, 2))))
			M[:, i, :, :] = A_solve.dot(b).reshape((nbs, 3, 2))
		M += MB_0

		return np.transpose(M, (0, 1, 3, 2)) 

	def face_reg(self, s_lst, b0, max_iters = 10):
		ns = len(s_lst)
		gamma = 500
		beta = 0.5

		bias = self.bs[1:] - self.bs[0]
		coef_ini = self.est_bs_coef(b0, bias, s_lst, np.zeros((ns, nbs), dtype = np.float32), gamma = gamma)
		coef_lst = coef_ini.copy()

		for i in range(max_iters):
			print('iters:', i, beta - 0.045 * i, gamma - 50 * i)
			M_lst = self.update_bs(b0, s_lst, coef_lst, self.faces, beta - 0.045 * i)
			bs_update = []
			for j in range(len(M_lst)):
				con_ind = np.argsort(np.sqrt(np.sum((self.bs[j + 1] - self.bs[0]) ** 2, axis = 1)))[:100]
				bs_update.append(self.recon_mesh(M_lst[j], self.faces, vn, con_ind, b0[con_ind]))
			bs_update = np.array(bs_update)

			bias = bs_update - b0
			coef_lst = self.est_bs_coef(b0, bias, s_lst, coef_ini, gamma = gamma - 50 * i)
		return bs_update


if __name__ == '__main__':
	from os.path import exists
	tester = ExampleFacialRig()
	from obj_render import GL_ObjRenderer 
	renderer = GL_ObjRenderer()

	region_ind = np.loadtxt('../region/region_in_head.txt', dtype = np.int32)
	ids = np.loadtxt('../test_data/test_people_ids.txt', dtype = np.int32)
	exp_names = open('../test_data/exp_names.txt', 'r+').readlines()

	bias = tester.bs[1:] - tester.bs[0]

	for i in ids:
		root = '../output/ebfr/' + str(i) + '/'

		n_head = TriMesh()
		n_head.load('../output/inmouth/' + str(i) + '.obj')
		n_head.del_by_vert_ind(region_ind)
		n_head_img = renderer.render_mesh_list_multi_view([n_head], angles = [0])[0]

		s_lst = []
		for line in exp_names:
			exp = line.strip('\n')
			if not exists(root + 'reg/' + exp + '_region.obj'):
				continue
			mesh = TriMesh()
			mesh.load(root + 'reg/' + exp + '_region.obj')
			s_lst.append(mesh.vertices)

		bs_update = tester.face_reg(s_lst, n_head.vertices)
		for i in range(len(bs_update)):
			print('bs:', i)
			mesh = TriMesh()
			mesh.faces = tester.faces
			mesh.vertices = bs_update[i]
			mesh.save(root + 'ebfr_bs/' + str(i + 1) + '.obj')

			img = renderer.render_mesh_list_multi_view([mesh], angles = [0])[0]
			temp_img = cv.imread('../blendshape/bfm/' + str(i + 1) + '.jpg')
			transfer_img = cv.imread(root + 'transfer_bs/' + str(i + 1) + '.jpg')

			cv.imwrite(root + 'ebfr_bs/' + str(i + 1) + '.jpg', \
				np.hstack((temp_img, transfer_img, img, n_head_img)))
