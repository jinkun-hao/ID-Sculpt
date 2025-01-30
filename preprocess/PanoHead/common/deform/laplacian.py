import numpy as np 
import scipy.sparse as sp

from common.mesh import TriMesh 

class Laplacian:
	def __init__(self):
		pass 

	def laplacian_operator(self, mesh):
		vnum = len(mesh.vertices)
		fnum = len(mesh.faces)
		laplacian = np.zeros((vnum, vnum), dtype = np.float32)
		edge_share_num = np.zeros((vnum, vnum), dtype = np.uint8)

		for i in range(3):
			vec_i_i1 = mesh.vertices[mesh.faces[:, (i + 1) % 3]] - mesh.vertices[mesh.faces[:, i]]
			vec_i_i2 = mesh.vertices[mesh.faces[:, (i + 2) % 3]] - mesh.vertices[mesh.faces[:, i]]
			vec_len_i_i1 = np.sqrt(np.sum(vec_i_i1 ** 2, axis = 1)) 
			vec_len_i_i2 = np.sqrt(np.sum(vec_i_i2 ** 2, axis = 1))
			vec_i_i1 = vec_i_i1 / (vec_len_i_i1[:, np.newaxis] + 1e-12)
			vec_i_i2 = vec_i_i2 / (vec_len_i_i2[:, np.newaxis] + 1e-12)

			cos_i = np.sum(vec_i_i1 * vec_i_i2, axis = 1)
			tan_half_theta = np.sqrt((1 - cos_i) / (1 + cos_i))

			for j in range(mesh.face_num()):
				laplacian[mesh.faces[j, i], mesh.faces[j, (i + 1) % 3]] += tan_half_theta / vec_len_i_i1 
				laplacian[mesh.faces[j, i], mesh.faces[j, (i + 2) % 3]] += tan_half_theta / vec_len_i_i2

				edge_share_num[mesh.faces[j, i], mesh.faces[j, (i + 1) % 3]] += 1 
				edge_share_num[mesh.faces[j, i], mesh.faces[j, (i + 2) % 3]] += 1

		connect = np.nonzero(edge_share_num)
		laplacian[connect] /= edge_share_num[connect]
		laplacian /= -np.sum(laplacian, axis = 1)[:, np.newaxis]
		laplacian += np.eye(vnum)

		return sp.coo_matrix(laplacian)

	def laplacian_operator_sparse(self, mesh):
		vnum = len(mesh.vertices)
		fnum = len(mesh.faces)

		edge_data = np.array([]).astype(np.uint8)
		data = np.array([]).astype(np.float32)
		row = np.array([]).astype(np.uint16)
		col = np.array([]).astype(np.uint16)

		for i in range(3):
			vec_i_i1 = mesh.vertices[mesh.faces[:, (i + 1) % 3]] - mesh.vertices[mesh.faces[:, i]]
			vec_i_i2 = mesh.vertices[mesh.faces[:, (i + 2) % 3]] - mesh.vertices[mesh.faces[:, i]]
			vec_len_i_i1 = np.sqrt(np.sum(vec_i_i1 ** 2, axis = 1)) 
			vec_len_i_i2 = np.sqrt(np.sum(vec_i_i2 ** 2, axis = 1))
			vec_i_i1 = vec_i_i1 / (vec_len_i_i1[:, np.newaxis] + 1e-12)
			vec_i_i2 = vec_i_i2 / (vec_len_i_i2[:, np.newaxis] + 1e-12)

			cos_i = np.sum(vec_i_i1 * vec_i_i2, axis = 1)
			tan_half_theta = np.sqrt((1 - cos_i) / (1 + cos_i))

			row = np.hstack((row, np.tile(mesh.faces[:, i], 2)))
			col = np.hstack((col, mesh.faces[:, (i + 1) % 3], mesh.faces[:, (i + 2) % 3]))

			data = np.hstack((data, tan_half_theta / vec_len_i_i1, tan_half_theta / vec_len_i_i2))
			edge_data = np.hstack((edge_data, np.ones(fnum * 2, dtype = np.uint8)))

		laplacian = sp.coo_matrix((data, (row, col)), shape = (vnum, vnum))
		laplacian.sum_duplicates()
		edge_share_num = sp.coo_matrix((edge_data.astype(np.float32), (row, col)), shape = (vnum, vnum))
		edge_share_num.sum_duplicates()

		assert((laplacian.row == edge_share_num.row).all() and (laplacian.col == edge_share_num.col).all())
		laplacian.data = laplacian.data / edge_share_num.data

		vert_weight_sum = np.asarray(-laplacian.sum(axis = 1)).reshape(-1)
		for i in range(len(laplacian.data)):
			laplacian.data[i] /= vert_weight_sum[laplacian.row[i]]

		laplacian.data = np.hstack((laplacian.data, np.ones(vnum)))
		laplacian.row = np.hstack((laplacian.row, np.arange(vnum)))
		laplacian.col = np.hstack((laplacian.col, np.arange(vnum)))

		laplacian.sum_duplicates()

		return laplacian

	def deform(self, mesh, c_vert_lst, c_idx_lst, c_weight_lst):
		A = self.laplacian_operator_sparse(mesh)
		b = A.dot(mesh.vertices)

		for i in range(len(c_vert_lst)):
			if len(c_vert_lst[i]) == 0:
				continue
			if not type(c_weight_lst[i]) is np.ndarray:
				c_weight = np.full(len(c_idx_lst[i]), c_weight_lst[i])
			else:
				c_weight = c_weight_lst[i]
				
			c = sp.coo_matrix((c_weight, \
				(np.arange(len(c_idx_lst[i])).astype(np.uint16), c_idx_lst[i].astype(np.uint16))), \
				shape = (len(c_idx_lst[i]), mesh.vert_num()))

			A = sp.vstack((A, c))
			b = np.vstack((b, c_vert_lst[i] * c_weight[:, np.newaxis]))

		mesh_copy = mesh.copy()
		mesh_copy.vertices = sp.linalg.spsolve(A.T.dot(A), A.T.dot(b))
		return mesh_copy

	def fuse(self, mesh, region_verts, region_idx, c_verts_lst, c_idx_lst, seam_weight = 1, c_weight_lst = None):

		pos_mesh = TriMesh(vertices = mesh.vertices.copy(), faces = mesh.faces.copy())
		pos_mesh.del_by_vert_ind(region_idx)

		seam_vert_0, seam_vert_1 = pos_mesh.find_edge_verts()
		seam_in_pos = np.array(list(set(seam_vert_0).union(set(seam_vert_1))))
		seam_in_mesh = region_idx[seam_in_pos]

		neg_mesh = TriMesh(vertices = mesh.vertices.copy(), faces = mesh.faces.copy())
		neg_vert_mask = np.full(mesh.vert_num(), True)
		neg_vert_mask[region_idx] = False
		neg_vert_mask[seam_in_mesh] = True 
		neg_mesh.del_by_vert_mask(neg_vert_mask)

		indexer = np.full(mesh.vert_num(), -1, dtype = np.int)
		indexer[neg_vert_mask] = np.arange(neg_mesh.vert_num())
		seam_in_neg = indexer[seam_in_mesh]

		A_laplacian = self.laplacian_operator_sparse(neg_mesh)
		sigma = A_laplacian.dot(neg_mesh.vertices)

		A_seam = sp.coo_matrix(( \
			np.full(len(seam_in_neg), seam_weight, dtype = np.float32), \
			(np.arange(len(seam_in_neg)).astype(np.uint16), seam_in_neg.astype(np.uint16))), \
		    shape = (len(seam_in_neg), neg_mesh.vert_num()))

		A = sp.vstack((A_laplacian, A_seam))
		b = np.vstack((sigma, \
			region_verts[seam_in_pos] * seam_weight))

		for i in range(len(c_verts_lst)):
			if c_weight_lst is None:
				c_weight = 1 
			else:
				c_weight = c_weight_lst[i]

			c_in_neg = indexer[c_idx_lst[i]]
			A_constrain = sp.coo_matrix( \
				(np.full(len(c_idx_lst[i]), c_weight, dtype = np.float32),\
				(np.arange(len(c_idx_lst[i])).astype(np.uint16), c_in_neg.astype(np.uint16))), \
				shape = (len(c_idx_lst[i]), neg_mesh.vert_num()))

			A = sp.vstack((A, A_constrain))
			b = np.vstack((b, c_verts_lst[i] * c_weight))
		
		neg_new_verts = sp.linalg.spsolve(A.T.dot(A), A.T.dot(b))

		tmp_mesh = mesh.copy()
		tmp_mesh.vertices[neg_vert_mask] = neg_new_verts
		tmp_mesh.vertices[region_idx] = region_verts

		return tmp_mesh

	def deform_wper_proj(self, verts_prev, lap_operator, lap_res, cor_data, cons_3d, cons_2d, ws):
		w_cor, w_cor_plane, w_p3d, w_p2d = ws
		vn = len(verts_prev)

		A = sp.block_diag((lap_operator, lap_operator, lap_operator))
		b = lap_res.T.reshape((-1, 1))

		if not cons_3d is None:
			p3d_ind, p3d_align, p3d_weights = cons_3d
			n_p3d = len(p3d_ind)

			row = np.arange(n_p3d * 3)
			col = np.column_stack((p3d_ind, p3d_ind + vn, p3d_ind + vn * 2)).flatten()
			data = p3d_weights.repeat(3) * w_p3d
			C = sp.coo_matrix((data, (row, col)), shape = (n_p3d * 3, vn * 3))
			A = sp.vstack((A, C))
			b = np.vstack((b, (p3d_align * p3d_weights[:, np.newaxis] * w_p3d).reshape((-1, 1)) ))

		if not cor_data is None:
			src_ind, cor_verts, cor_normal, cor_weights = cor_data 
			n_cor = len(src_ind)

			row = np.arange(n_cor * 3)
			col = np.column_stack((src_ind, src_ind + vn, src_ind + vn * 2)).flatten()
			data = cor_weights.repeat(3) * w_cor
			C = sp.coo_matrix((data, (row, col)), shape = (n_cor * 3, vn * 3))
			A = sp.vstack((A, C))
			b = np.vstack((b, (cor_verts * cor_weights[:, np.newaxis] * w_cor).reshape((-1, 1)) ))

			row = np.arange(n_cor).repeat(3)
			col = np.column_stack((src_ind, src_ind + vn, src_ind + vn * 2)).flatten()
			data = cor_normal * cor_weights[:, np.newaxis] * w_cor_plane
			C = sp.coo_matrix((data.flatten().astype(np.float32), (row, col)), shape = (n_cor, vn * 3))
			A = sp.vstack((A, C))
			b = np.vstack((b, (np.sum(cor_verts * cor_normal, axis = 1) * cor_weights * w_cor_plane).reshape((-1, 1))))

		p2d_ind, p2d_lst, cam_mat_lst, trans_mat_lst, p2d_weights = cons_2d
		n_p2d = len(p2d_ind) 

		for i in range(len(p2d_lst)):
			trans_mat = trans_mat_lst[i]
			cam_mat = cam_mat_lst[i]
			p = p2d_lst[i]
			z_hat = np.sum(trans_mat[2, :3][np.newaxis, :] * verts_prev[p2d_ind], axis = 1) + trans_mat[2, 3]

			c0 = np.array([cam_mat[0, 0] * trans_mat[0, 0] + cam_mat[0, 1] * trans_mat[1, 0], \
				cam_mat[0, 0] * trans_mat[0, 1] + cam_mat[0, 1] * trans_mat[1, 1], \
				cam_mat[0, 0] * trans_mat[0, 2] + cam_mat[0, 1] * trans_mat[1, 2]])
			c0 = np.tile(c0, n_p2d).reshape((-1, 3)) / z_hat[:, np.newaxis]

			c1 = np.array([cam_mat[1, 0] * trans_mat[0, 0] + cam_mat[1, 1] * trans_mat[1, 0], \
				cam_mat[1, 0] * trans_mat[0, 1] + cam_mat[1, 1] * trans_mat[1, 1], \
				cam_mat[1, 0] * trans_mat[0, 2] + cam_mat[1, 1] * trans_mat[1, 2]])
			c1 = np.tile(c1, n_p2d).reshape((-1, 3)) / z_hat[:, np.newaxis]

			data = np.hstack((c0, c1)) * p2d_weights[:, np.newaxis] * w_p2d
			row = np.arange(n_p2d * 2).repeat(3)
			col = np.column_stack((p2d_ind.repeat(2), p2d_ind.repeat(2) + vn, p2d_ind.repeat(2) + vn * 2)).flatten()
			Ci = sp.coo_matrix((data.flatten(), (row, col)), shape = (n_p2d * 2, vn * 3))
			A = sp.vstack((A, Ci))

			bi_0 = (p[:, 0] - cam_mat[0, 2]) - cam_mat[0, 0] * trans_mat[0, 3] / z_hat - cam_mat[0, 1] * trans_mat[1, 3] / z_hat
			bi_1 = (p[:, 1] - cam_mat[1, 2]) - cam_mat[1, 0] * trans_mat[0, 3] / z_hat - cam_mat[1, 1] * trans_mat[1, 3] / z_hat
			bi = np.column_stack((bi_0, bi_1)) * p2d_weights[:, np.newaxis] * w_p2d
			b = np.vstack((b, bi.reshape((-1, 1))))

		# verts = sp.linalg.spsolve(A.T.dot(A), A.T.dot(b)).reshape((3, -1)).T
		x, flag = sp.linalg.cg(A.T.dot(A), A.T.dot(b), x0 = verts_prev.T.flatten(), tol = 1e-6)
		verts = x.reshape((3, -1)).T

		return verts