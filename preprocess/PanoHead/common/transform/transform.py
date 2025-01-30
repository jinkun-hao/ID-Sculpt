import numpy as np 
from math import sin, cos, pi, asin, atan2, acos, sqrt

def transform_ortho(vert, R, s, t):
	return R.dot(vert.T).T * s + t

def rotate_angle(v1, v2):
	cro = np.cross(v1, v2)
	cro_norm = np.sqrt(np.sum(cro ** 2, axis = 1))

	angle = np.arctan2(cro_norm, np.sum(v1 * v2, axis = 1))
	mask = cro[:, 2] < 0 
	angle[mask] = 2 * pi - angle[mask]

	return angle

def transform_proj_ortho(vert, R, s, t):
	return R.dot(vert.T)[:2].T * s + t[:2]

def project_pers(vert, trans_mat, cam_mat):
	trans = np.hstack((vert, np.ones((len(vert), 1)))).dot(trans_mat.T)
	proj = trans[:, :3].dot(cam_mat.T)
	proj[:, :2] /= proj[:, 2][:, np.newaxis]
	return proj[:, :2]

def sRt_vec_to_mat(pose):
	mat = np.eye(4, dtype = np.float32)
	mat[:3, :3] = np.reshape(pose[1:10], (3, 3)) * pose[0]
	mat[:3, 3] = pose[10:]
	return mat

def rad2angle(rad):
	return rad / pi * 180

def angle2rad(angle):
	return angle / 180 * pi

def euler_angle_rotate_mat(r):
	if r[1, 0] < 1:
		if r[1, 0] > -1:
			radz = asin(r[1, 0])
			rady = atan2(-r[2, 0], r[0, 0])
			radx = atan2(-r[1, 2], r[1, 1])
		else:
			radz = -pi / 2
			rady = -atan2(r[2, 1], r[2, 2])
			radx = 0
	else:
		radz = pi / 2
		rady = atan2(r[2, 1], r[2, 2])
		radz = 0
	return radx, rady, radz 

def rotate_mat_euler_angle(radx, rady, radz):
	matx = np.eye(3, 3)
	maty = np.eye(3, 3)
	matz = np.eye(3, 3)
	matx[1, 1] = cos(radx)
	matx[2, 2] = cos(radx)
	matx[1, 2] = -sin(radx)
	matx[2, 1] = sin(radx)

	maty[0, 0] = cos(rady)
	maty[2, 2] = cos(rady)
	maty[0, 2] = sin(rady)
	maty[2, 0] = -sin(rady)

	matz[0, 0] = cos(radz)
	matz[1, 1] = cos(radz)
	matz[1, 0] = sin(radz)
	matz[0, 1] = -sin(radz)

	return maty.dot(matz).dot(matx)

def rodrigues(axis, theta):
	u = axis[:, np.newaxis].dot(axis[np.newaxis, :])

	u_ = np.zeros((3, 3), dtype = np.float32)
	u_[0, 1] = -axis[2]
	u_[0, 2] = axis[1]
	u_[1, 0] = axis[2]
	u_[1, 2] = -axis[0]
	u_[2, 0] = -axis[1]
	u_[2, 1] = axis[0]

	r = cos(theta) * np.eye(3, dtype = np.float32) + sin(theta) * u_ + (1 - cos(theta)) * u 
	return r 

def quat_rotation_mat(q):
	r = np.zeros((3, 3), dtype = np.float32)
	r[0, 0] = 1 - 2 * q[2] ** 2 - 2 * q[3] ** 2 
	r[0, 1] = 2 * q[1] * q[2] - 2 * q[0] * q[3]
	r[0, 2] = 2 * q[1] * q[3] + 2 * q[0] * q[2]

	r[1, 0] = 2 * q[1] * q[2] + 2 * q[0] * q[3]
	r[1, 1] = 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2 
	r[1, 2] = 2 * q[2] * q[3] - 2 * q[0] * q[1]

	r[2, 0] = 2 * q[1] * q[3] - 2 * q[0] * q[2]
	r[2, 1] = 2 * q[2] * q[3] + 2 * q[0] * q[1]
	r[2, 2] = 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2
	return r

def rot_mat_to_quat(r):
	q = np.zeros(4, dtype = np.float32)
	q[0] = sqrt(r[0, 0] + r[1, 1] + r[2, 2] + 1) / 2
	q[1] = (r[1, 2] - r[2, 1]) / 4 / q0
	q[2] = (r[2, 0] - r[0, 2]) / 4 / q0 
	q[3] = (r[0, 1] - r[1, 0]) / 4 / q0

	return q 

def quat_mul(qa, qb):
	res = np.zeros(4, dtype = np.float32)
	res[0] = qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3]
	res[1] = qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2]
	res[2] = qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1]
	res[3] = qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0]

	return res 

def quat_inv(q):
	res = q.copy()
	res[1:] *= -1 
	res /= (q ** 2).sum()

def quat_to_axis_angle(q):
	theta = 2 * acos(q0)
	axis = q[1:] / sqrt(1 - q[0] ** 2)
	return theta, axis

def quat_angle(q0, q1):
	return acos((q0 * q1).sum())

# def quat_slerp(q0, q1, t):