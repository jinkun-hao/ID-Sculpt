import numpy as np 
import scipy.sparse as sp

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.autoinit import context
import pycuda.gpuarray as gpuarray
import skcuda.linalg

from common.mesh import TriMesh 

class EmbeddedDeform:
	def __init__(self):
		pass

	def compute_con(self, vertices, cons, v2n, nodes, r_lst, t_lst, sn = 4):
		verts_norm = vertices[:, np.newaxis, :] - nodes[v2n]
		dist = np.sqrt(np.sum(verts_norm ** 2, axis = 2))
		weights = (1 - dist[:, :sn] / (dist[:, sn][:, np.newaxis])) ** 2
		weights = weights / (np.sum(weights, axis = 1)[:, np.newaxis] + 1e-15)

		f_con = np.sum(weights[:, :, np.newaxis] * \
			(np.matmul(r_lst[v2n[:, :sn]], verts_norm[:, :sn, :, np.newaxis]).squeeze(axis = 3) + nodes[v2n[:, :sn]] + t_lst[v2n[:, :sn]]), axis = 1) \
			- cons
		f_con = f_con.flatten()

		vn, nn = len(cons), len(r_lst)
		j_con = np.zeros((vn * 3, nn * 12), dtype = np.float32)
		for i in range(sn):
			rcol_ind = (v2n[:, i].repeat(3).reshape((-1, 3)) * 9 + np.array([[0, 1, 2]])).flatten()
			for j in range(3):
				row_ind =  np.arange(vn).repeat(3) * 3 + j
				col_ind =  rcol_ind + 3 * j
				j_con[row_ind, col_ind] = (verts_norm[:, i, :] * weights[:, i][:, np.newaxis]).flatten()

				row_ind = np.arange(vn) * 3 + j
				col_ind = nn * 9 + v2n[:, i] * 3 + j
				j_con[row_ind, col_ind] = weights[:, i]

		return f_con, j_con

	def compute_reg(self, nodes, graph, r_lst, t_lst):
		n2n_row, n2n_col = np.where(graph)
		mask = n2n_row < n2n_col 
		n2n = np.column_stack((n2n_row[mask], n2n_col[mask]))

		f_reg = np.matmul(r_lst[n2n[:, 1]], (nodes[n2n[:, 0]] - nodes[n2n[:, 1]])[:, :, np.newaxis]).squeeze() \
			+ nodes[n2n[:, 1]] + t_lst[n2n[:, 1]] - nodes[n2n[:, 0]] - t_lst[n2n[:, 0]]
		f_reg = f_reg.flatten()
		
		j_reg = np.zeros((len(n2n) * 3, len(r_lst) * 12), dtype = np.float32)
		for i in range(3):
			row_ind = np.arange(len(n2n)) * 3 + i
			col_ind = len(r_lst) * 9 + n2n[:, 1] * 3 + i 
			j_reg[row_ind, col_ind] = 1 

			col_ind = len(r_lst) * 9 + n2n[:, 0] * 3 + i 
			j_reg[row_ind, col_ind] = -1 

			row_ind = np.arange(len(n2n)).repeat(3) * 3 + i 
			col_ind = n2n[:, 1].repeat(3).reshape((-1, 3)) * 9 + np.array([[0, 1, 2]]) + 3 * i
			j_reg[row_ind, col_ind.flatten()] = (nodes[n2n[:, 0]] - nodes[n2n[:, 1]]).flatten()

		return f_reg, j_reg

	def compute_rot(self, r_lst):
		nn = len(r_lst)
		f0 = np.sum(r_lst[:, :, 0] * r_lst[:, :, 1], axis = 1)
		f1 = np.sum(r_lst[:, :, 0] * r_lst[:, :, 2], axis = 1)
		f2 = np.sum(r_lst[:, :, 1] * r_lst[:, :, 2], axis = 1)

		g0 = np.sum(r_lst[:, :, 0] ** 2, axis = 1)
		g1 = np.sum(r_lst[:, :, 1] ** 2, axis = 1)
		g2 = np.sum(r_lst[:, :, 2] ** 2, axis = 1)

		f_rot = np.column_stack((g0 - 1, g1 - 1, g2 - 1, f0, f1, f2)).flatten()

		j_rot = np.zeros((nn * 6, nn * 12), dtype = np.float32)

		row_ind = np.arange(nn).repeat(3) * 6
		col_ind = np.arange(nn).repeat(3).reshape((-1, 3)) * 9
		j_rot[row_ind + 0, (col_ind + np.array([0, 3, 6])).flatten()] = 2 * r_lst[:, :, 0].flatten()
		j_rot[row_ind + 1, (col_ind + np.array([1, 4, 7])).flatten()] = 2 * r_lst[:, :, 1].flatten()
		j_rot[row_ind + 2, (col_ind + np.array([2, 5, 8])).flatten()] = 2 * r_lst[:, :, 2].flatten()

		j_rot[row_ind + 3, (col_ind + np.array([0, 3, 6])).flatten()] = r_lst[:, :, 1].flatten()
		j_rot[row_ind + 3, (col_ind + np.array([1, 4, 7])).flatten()] = r_lst[:, :, 0].flatten()

		j_rot[row_ind + 4, (col_ind + np.array([0, 3, 6])).flatten()] = r_lst[:, :, 2].flatten()
		j_rot[row_ind + 4, (col_ind + np.array([2, 5, 8])).flatten()] = r_lst[:, :, 0].flatten()

		j_rot[row_ind + 5, (col_ind + np.array([1, 4, 7])).flatten()] = r_lst[:, :, 2].flatten()
		j_rot[row_ind + 5, (col_ind + np.array([2, 5, 8])).flatten()] = r_lst[:, :, 1].flatten()

		return f_rot, j_rot 

	def compute_con_sparse(self, vertices, cons, v2n, nodes, weights, con_weights, r_lst, t_lst, sn = 4):
		verts_norm = vertices[:, np.newaxis, :] - nodes[v2n[:, :sn]]
		f_con = np.sum(weights[:, :, np.newaxis] * \
			(np.matmul(r_lst[v2n[:, :sn]], verts_norm[:, :, :, np.newaxis]).squeeze(axis = 3) + nodes[v2n[:, :sn]] + t_lst[v2n[:, :sn]]), axis = 1) \
			- cons
		f_con = f_con * con_weights[:, np.newaxis]

		vn, nn = len(cons), len(r_lst)
		row, col = np.array([], dtype = np.int32), np.array([], dtype = np.int32)
		data = np.array([], dtype = np.float32)
		for i in range(sn):
			rcol_ind = (v2n[:, i].repeat(3).reshape((-1, 3)) * 9 + np.array([[0, 1, 2]], dtype = np.int32)).flatten()
			for j in range(3):
				row = np.append(row, np.arange(vn).repeat(3) * 3 + j)
				col = np.append(col, rcol_ind + 3 * j)
				data = np.append(data, (verts_norm[:, i, :] * weights[:, i][:, np.newaxis] * con_weights[:, np.newaxis]).flatten())

				row = np.append(row, np.arange(vn) * 3 + j)
				col = np.append(col, nn * 9 + v2n[:, i] * 3 + j)
				data = np.append(data, weights[:, i] * con_weights)

		j_con = sp.coo_matrix((data, (row, col)), shape = (vn * 3, nn * 12))
		j_con.sum_duplicates()

		return f_con.flatten(), j_con

	def compute_con_plane_sparse(self, vertices, cons, normal, v2n, nodes, weights, con_weights, r_lst, t_lst, sn = 4):
		verts_norm = vertices[:, np.newaxis, :] - nodes[v2n[:, :sn]]
		f_con = np.sum(weights[:, :, np.newaxis] * \
			(np.matmul(r_lst[v2n[:, :sn]], verts_norm[:, :, :, np.newaxis]).squeeze(axis = 3) + nodes[v2n[:, :sn]] + t_lst[v2n[:, :sn]]), axis = 1) \
			- cons
		f_con_plane = np.sum(f_con * normal, axis = 1) * con_weights

		vn, nn = len(cons), len(r_lst)
		row, col = np.array([], dtype = np.int32), np.array([], dtype = np.int32)
		data = np.array([], dtype = np.float32)
		for i in range(sn):
			for j in range(3):
				row = np.append(row, np.arange(vn))
				col = np.append(col, nn * 9 + v2n[:, i] * 3 + j)
				data = np.append(data, weights[:, i] * con_weights * normal[:, j])

				row = np.append(row, np.arange(vn).repeat(3))
				col = np.append(col, (v2n[:, i].repeat(3).reshape((-1, 3)) * 9 + np.array([0, 1, 2], dtype = np.int32) + j * 3).flatten())
				data = np.append(data, (verts_norm[:, i, :] * (weights[:, i] * con_weights * normal[:, j])[:, np.newaxis]).flatten())

		j_con_plane = sp.coo_matrix((data, (row, col)), shape = (vn, nn * 12))
		j_con_plane.sum_duplicates()

		return f_con_plane.flatten(), j_con_plane

	def compute_reg_sparse(self, r_lst, t_lst, nodes, n2n):		
		f_reg = np.matmul(r_lst[n2n[:, 1]], (nodes[n2n[:, 0]] - nodes[n2n[:, 1]])[:, :, np.newaxis]).squeeze() \
			+ nodes[n2n[:, 1]] + t_lst[n2n[:, 1]] - nodes[n2n[:, 0]] - t_lst[n2n[:, 0]]

		row, col = np.array([], dtype = np.int32), np.array([], dtype = np.int32)
		data = np.array([], dtype = np.float32)
		for i in range(3):
			row = np.append(row, np.arange(len(n2n)) * 3 + i)
			col = np.append(col, len(r_lst) * 9 + n2n[:, 1] * 3 + i)
			data = np.append(data, np.ones(len(n2n), dtype = np.float32))

			row = np.append(row, np.arange(len(n2n)) * 3 + i)
			col = np.append(col, len(r_lst) * 9 + n2n[:, 0] * 3 + i)
			data = np.append(data, np.ones(len(n2n), dtype = np.float32) * -1)

			row = np.append(row, np.arange(len(n2n)).repeat(3) * 3 + i)
			col = np.append(col, (n2n[:, 1].repeat(3).reshape((-1, 3)) * 9 + np.array([[0, 1, 2]], dtype = np.int32) + 3 * i).flatten())
			data = np.append(data, (nodes[n2n[:, 0]] - nodes[n2n[:, 1]]).flatten())

		j_reg = sp.coo_matrix((data, (row, col)), shape = (len(n2n) * 3, len(r_lst) * 12))
		j_reg.sum_duplicates()
		return f_reg.flatten(), j_reg

	def compute_rot_sparse(self, r_lst):
		nn = len(r_lst)
		f0 = np.sum(r_lst[:, :, 0] * r_lst[:, :, 1], axis = 1)
		f1 = np.sum(r_lst[:, :, 0] * r_lst[:, :, 2], axis = 1)
		f2 = np.sum(r_lst[:, :, 1] * r_lst[:, :, 2], axis = 1)

		g0 = np.sum(r_lst[:, :, 0] ** 2, axis = 1)
		g1 = np.sum(r_lst[:, :, 1] ** 2, axis = 1)
		g2 = np.sum(r_lst[:, :, 2] ** 2, axis = 1)

		f_rot = np.column_stack((g0 - 1, g1 - 1, g2 - 1, f0, f1, f2)).flatten()

		row, col = np.array([], dtype = np.int32), np.array([], dtype = np.int32)
		data = np.array([], dtype = np.float32)

		rind = np.arange(nn).repeat(3) * 6
		cind = np.arange(nn).repeat(3).reshape((-1, 3)) * 9

		row = np.append(row, np.hstack((rind + 0, rind + 1, rind + 2)))
		col = np.append(col, np.vstack((cind + np.array([0, 3, 6]), cind + np.array([1, 4, 7]), cind + np.array([2, 5, 8]))).flatten().astype(np.int32))
		data = np.append(data, np.hstack((2 * r_lst[:, :, 0].flatten(), 2 * r_lst[:, :, 1].flatten(), 2 * r_lst[:, :, 2].flatten())))

		row = np.append(row, np.hstack((rind + 3, rind + 3)))
		col = np.append(col, np.vstack((cind + np.array([0, 3, 6]), cind + np.array([1, 4, 7]))).flatten().astype(np.int32))
		data = np.append(data, np.hstack((r_lst[:, :, 1].flatten(), r_lst[:, :, 0].flatten())))

		row = np.append(row, np.hstack((rind + 4, rind + 4)))
		col = np.append(col, np.vstack((cind + np.array([0, 3, 6]), cind + np.array([2, 5, 8]))).flatten().astype(np.int32))
		data = np.append(data, np.hstack((r_lst[:, :, 2].flatten(), r_lst[:, :, 0].flatten())))

		row = np.append(row, np.hstack((rind + 5, rind + 5)))
		col = np.append(col, np.vstack((cind + np.array([1, 4, 7]), cind + np.array([2, 5, 8]))).flatten().astype(np.int32))
		data = np.append(data, np.hstack((r_lst[:, :, 2].flatten(), r_lst[:, :, 1].flatten())))

		j_rot = sp.coo_matrix((data, (row, col)), shape = (nn * 6, nn * 12))
		j_rot.sum_duplicates()
		return f_rot, j_rot

	def compute_con_wper_sparse(self, vertices, prev_deform, lmk_2d_lst, cam_mat_lst, trans_mat_lst, v2n, nodes, weights, con_weights, r_lst, t_lst, sn = 4):
		verts_norm = vertices[:, np.newaxis, :] - nodes[v2n[:, :sn]]
		verts_def = np.sum(weights[:, :, np.newaxis] * \
			(np.matmul(r_lst[v2n[:, :sn]], verts_norm[:, :, :, np.newaxis]).squeeze(axis = 3) + nodes[v2n[:, :sn]] + t_lst[v2n[:, :sn]]), axis = 1)

		pn, vn, nn = len(lmk_2d_lst), len(lmk_2d_lst[0]), len(r_lst)
		
		row, col = np.array([], dtype = np.int32), np.array([], dtype = np.int32)
		data = np.array([], dtype = np.float32)
		f_proj = np.array([], dtype = np.float32)

		for i in range(pn):
			K = cam_mat_lst[i]
			T = trans_mat_lst[i]
			z_hat = np.sum(T[2, :3][np.newaxis, :] * prev_deform, axis = 1) + T[2, 3]
			p2d = lmk_2d_lst[i]

			c0 = np.array([K[0, 0] * T[0, 0] + K[0, 1] * T[1, 0], K[0, 0] * T[0, 1] + K[0, 1] * T[1, 1], K[0, 0] * T[0, 2] + K[0, 1] * T[1, 2]])
			c1 = np.array([K[1, 0] * T[0, 0] + K[1, 1] * T[1, 0], K[1, 0] * T[0, 1] + K[1, 1] * T[1, 1], K[1, 0] * T[0, 2] + K[1, 1] * T[1, 2]])

			proj_0 = (np.sum(c0 * verts_def, axis = 1) + K[0, 0] * T[0, 3] + K[0, 1] * T[1, 3]) / z_hat + K[0, 2]
			proj_1 = (np.sum(c1 * verts_def, axis = 1) + K[1, 0] * T[0, 3] + K[1, 1] * T[1, 3]) / z_hat + K[1, 2]

			f_proj = np.append(f_proj, ((np.column_stack((proj_0, proj_1)) - p2d) * con_weights[:, np.newaxis]).flatten())

			for j in range(sn):
				for k in range(3):
					u_deriv_r = weights[:, j][:, np.newaxis] * verts_norm[:, j, :] * c0[k] / z_hat[:, np.newaxis] * con_weights[:, np.newaxis]
					v_deriv_r = weights[:, j][:, np.newaxis] * verts_norm[:, j, :] * c1[k] / z_hat[:, np.newaxis] * con_weights[:, np.newaxis]

					u_deriv_t = weights[:, j] * c0[k] / z_hat * con_weights
					v_deriv_t = weights[:, j] * c1[k] / z_hat * con_weights

					data = np.append(data, np.hstack((u_deriv_r, u_deriv_t[:, np.newaxis], v_deriv_r, v_deriv_t[:, np.newaxis])).flatten())

					rind = (np.arange(vn) * 2 + i * vn * 2).repeat(4).reshape((-1, 4))
					row = np.append(row, np.hstack((rind, rind + 1)).flatten())

					cind_r = ((v2n[:, j] * 9 + k * 3).repeat(3).reshape((-1, 3)) + np.array([0, 1, 2], dtype = np.int32))
					cind_t = nn * 9 + v2n[:, j] * 3 + k
					cind = np.hstack((cind_r, cind_t[:, np.newaxis]))
					col = np.append(col, np.hstack((cind, cind)).flatten())

		j_proj = sp.coo_matrix((data, (row, col)), shape = (pn * 2 * vn, nn * 12))
		j_proj.sum_duplicates()

		return f_proj, j_proj

	def transform_verts(self, r_lst, t_lst, verts, nodes, v2n, weights, sn = 4):
		trans = verts[:, np.newaxis, :] - nodes[v2n[:, :sn]]
		trans = np.matmul(r_lst[v2n[:, :sn]], trans[:, :, :, np.newaxis]).squeeze(axis = 3)
		trans = trans + nodes[v2n[:, :sn]] + t_lst[v2n[:, :sn]]
		trans = np.sum(weights[:, :, np.newaxis] * trans, axis = 1)

		return trans

	def solve_node_motion_p2p(self, x, src_ind, cor_verts, cor_weights, verts, nodes, v2n, n2n, weights,\
		w_reg = 1.0, w_rot = 10, w_con = 1, max_iters = 10):

		for i in range(max_iters):

			r_lst = x[0 : 9 * len(nodes)].reshape((-1, 3, 3))
			t_lst = x[9 * len(nodes):].reshape((-1, 3))

			f_rot, j_rot = self.compute_rot_sparse(r_lst)
			f_reg, j_reg = self.compute_reg_sparse(r_lst, t_lst, nodes, n2n)

			f_con, j_con = self.compute_con_sparse(verts[src_ind], cor_verts, \
				v2n[src_ind], nodes, weights[src_ind], cor_weights, r_lst, t_lst)

			J = sp.vstack((j_con * w_con, j_reg * w_reg, j_rot * w_rot))
			r = np.hstack((f_con * w_con, f_reg * w_reg, f_rot * w_rot))[:, np.newaxis]

			A_gpu = gpuarray.to_gpu(J.T.dot(J).todense().astype(np.float32))
			b_gpu = gpuarray.to_gpu(J.T.dot(r).astype(np.float32))
			skcuda.linalg.cho_solve(A_gpu, b_gpu)

			det = b_gpu.get()
			x = x - np.asarray(det).flatten()
			print(np.abs(det).mean())

			if np.abs(det).mean() < 5e-5:
				break
		return x

	def solve_node_motion(self, x, verts, nodes, v2n, n2n, weights, cor_data, p3d_data, p2d_data, prev_deform, ws, max_iters = 10):
		w_con, w_con_plane, w_reg, w_rot, w_con_p3d, w_con_p2d = ws

		src_ind, cor_verts, cor_normal, cor_weights = cor_data
		p3d_ind, p3d_align, p3d_weights = p3d_data
		if not p2d_data is None:
			p2d_ind, p2d_lst, cam_mat_lst, trans_mat_lst, p2d_weights = p2d_data

		for i in range(max_iters):

			r_lst = x[0 : 9 * len(nodes)].reshape((-1, 3, 3))
			t_lst = x[9 * len(nodes):].reshape((-1, 3))

			f_rot, j_rot = self.compute_rot_sparse(r_lst)
			f_reg, j_reg = self.compute_reg_sparse(r_lst, t_lst, nodes, n2n)

			f_con, j_con = self.compute_con_sparse(verts[src_ind], cor_verts, \
				v2n[src_ind], nodes, weights[src_ind], cor_weights, r_lst, t_lst)
			f_con_plane, j_con_plane = self.compute_con_plane_sparse(verts[src_ind], cor_verts, cor_normal, \
				v2n[src_ind], nodes, weights[src_ind], cor_weights, r_lst, t_lst)

			f_con_p3d, j_con_p3d = self.compute_con_sparse(verts[p3d_ind], p3d_align, \
				v2n[p3d_ind], nodes, weights[p3d_ind], p3d_weights, r_lst, t_lst)

			f_con_p2d, j_con_p2d = self.compute_con_wper_sparse(verts[p2d_ind], prev_deform[p2d_ind], \
				p2d_lst, cam_mat_lst, trans_mat_lst, v2n[p2d_ind], nodes, weights[p2d_ind], p2d_weights, r_lst, t_lst)

			J = sp.vstack((j_con * w_con, j_con_plane * w_con_plane, j_reg * w_reg, j_rot * w_rot, \
				j_con_p2d * w_con_p2d, j_con_p3d * w_con_p3d))
			r = np.hstack((f_con * w_con, f_con_plane * w_con_plane, f_reg * w_reg, f_rot * w_rot, \
				f_con_p2d * w_con_p2d, f_con_p3d * w_con_p3d))[:, np.newaxis]

			A_gpu = gpuarray.to_gpu(J.T.dot(J).todense().astype(np.float32))
			b_gpu = gpuarray.to_gpu(J.T.dot(r).astype(np.float32))
			skcuda.linalg.cho_solve(A_gpu, b_gpu)

			det = b_gpu.get()
			x = x - np.asarray(det).flatten()
			print(np.abs(det).mean())

			if np.abs(det).mean() < 5e-5:
				break
		return x