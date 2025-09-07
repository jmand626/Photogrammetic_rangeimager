import numpy as np
import math
from copy import deepcopy
from scipy.optimize import least_squares
from triangulation import estimate_RT_from_E, nonlinear_estimate_3d_point


class Frame:
    """
    Stores and manages data for a set of multiple views of the same scene.
    Initializes motion matrices, fundamental/essential matrices, and 3D structure.

    Args:
        matches (ndarray): Nx4 array of matching points (x1, y1, x2, y2)
        focal_length (float): focal length of the camera
        F (ndarray): fundamental matrix between the two views
        im_width (int): image width
        im_height (int): image height
    """
    def __init__(self, matches, focal_length, F, im_width, im_height):
        self.focal_length = focal_length
        self.im_height = im_height
        self.im_width = im_width
        self.matches = matches

        self.N = matches.shape[0]
        self.match_idx = np.array([
            np.arange(self.N), 
            np.arange(self.N, 2 * self.N)
        ])
        self.match_points = np.vstack((matches[:, :2], matches[:, 2:]))

        self.K = np.eye(3)
        self.K[0, 0] = self.K[1, 1] = focal_length
        self.E = self.K.T @ F @ self.K
        self.T = estimate_RT_from_E(self.E, matches.reshape((-1, 2, 2)), self.K)

        self.motion = np.zeros((2, 3, 4))
        self.motion[0, :, :-1] = np.eye(3)
        self.motion[1, :, :] = self.T

        self.structure = triangulate(self)


def neg_ones(size, dtype=np.int16):
    """
    Returns an array of the given size filled with -1.

    Args:
        size (tuple): shape of the output array
        dtype (type): desired dtype of the output array

    Returns:
        ndarray: array filled with -1
    """
    return -1 * np.ones(size, dtype=dtype)


def triangulate(frame):
    """
    Triangulates 3D points for all valid correspondences in a Frame object.

    Args:
        frame (Frame): contains camera parameters and matching points

    Returns:
        structure (ndarray): 3D coordinates (N, 3)
    """
    num_cameras, num_points = frame.match_idx.shape
    structure = np.zeros((num_points, 3))
    all_camera_matrices = np.zeros((num_cameras, 3, 4))

    for i in range(num_cameras):
        all_camera_matrices[i] = frame.K @ frame.motion[i]

    for i in range(num_points):
        valid_cameras = np.where(frame.match_idx[:, i] >= 0)[0]
        camera_matrices = all_camera_matrices[valid_cameras]
        x = np.array([
            frame.match_points[frame.match_idx[c, i]]
            for c in valid_cameras
        ])
        structure[i] = nonlinear_estimate_3d_point(x, camera_matrices)

    return structure


def rotation_matrix_to_angle_axis(R):
    """
    Converts a rotation matrix to angle-axis format.

    Args:
        R (ndarray): 3x3 rotation matrix

    Returns:
        angle_axis (ndarray): 3-vector representing axis * angle
    """
    angle_axis = np.array([0.0]*3)
    angle_axis[0] = R[2, 1] - R[1, 2]
    angle_axis[1] = R[0, 2] - R[2, 0]
    angle_axis[2] = R[1, 0] - R[0, 1]
    
    cos_theta = min(max((R[0,0]+R[1,1]+R[2,2] - 1.0) / 2.0, -1.0), 1.0)
    sin_theta = min(np.sqrt((angle_axis**2).sum())/2, 1.0);

    theta = math.atan2(sin_theta, cos_theta)

    k_threshold = 1e-12
    if ((sin_theta > k_threshold) or (sin_theta < -k_threshold)):
        r = theta / (2.0 * sin_theta)
        angle_axis = angle_axis * r
        return angle_axis

    if cos_theta > 0:
        angle_axis = angle_axis / 2
        return angle_axis

    inv_one_minus_cos_theta = 1.0 / (1.0 - cos_theta)

    for i in range(3):
        angle_axis[i] = theta * math.sqrt((R[i,i] - cos_theta) 
            * inv_one_minus_cos_theta)
        if((sin_theta < 0 and angle_axis[i] > 0) or
            (sin_theta > 0 and angle_axis[i] < 0)):
            angle_axis[i] *= -1

    return angle_axis


def angle_axis_to_rotation_matrix(angle_axis):
    """
    Converts an angle-axis vector to a rotation matrix.

    Args:
        angle_axis (ndarray): 3D angle-axis vector

    Returns:
        R (ndarray): 3x3 rotation matrix
    """
    theta2 = np.dot(angle_axis, angle_axis)
    R = np.zeros((3,3))
    if theta2 > 0:
        theta = np.sqrt(theta2)
        wx, wy, wz = tuple(angle_axis / theta)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        R[0, 0] = cos_theta + wx * wx * (1 - cos_theta)
        R[1, 0] = wz * sin_theta + wx * wy * (1 - cos_theta)
        R[2, 0] = -wy * sin_theta + wx * wz * (1 - cos_theta)
        R[0, 1] =  wx * wy * (1 - cos_theta) - wz * sin_theta;
        R[1, 1] = cos_theta   + wy * wy * (1 - cos_theta);
        R[2, 1] =  wx * sin_theta   + wy * wz * (1 - cos_theta);
        R[0, 2] =  wy * sin_theta   + wx * wz * (1 - cos_theta);
        R[1, 2] = -wx * sin_theta   + wy * wz * (1 - cos_theta);
        R[2, 2] = cos_theta   + wz * wz * (1 - cos_theta);

    else:
        # At zero, we switch to using the first order Taylor expansion.
        R[0, 0] =  1
        R[1, 0] = -angle_axis[2]
        R[2, 0] =  angle_axis[1]
        R[0, 1] =  angle_axis[2]
        R[1, 1] =  1
        R[2, 1] = -angle_axis[0]
        R[0, 2] = -angle_axis[1]
        R[1, 2] =  angle_axis[0]
        R[2, 2] = 1
    return R


def cross_product_mat(a):
    """
    Returns the cross product matrix [a]_x.

    Args:
        a (ndarray): 3D vector

    Returns:
        ndarray: 3x3 cross-product matrix
    """
    m = np.zeros((3,3))
    m[1,0] = a[2]
    m[0,1] = -a[2]
    m[2,0] = -a[1]
    m[0,2] = a[1]
    m[2,1] = a[0]
    m[1,2] = -a[0]
    return m


def angle_axis_rotate(angle_axis, pt):
    """
    Applies an angle-axis rotation to a set of 3D points.

    Args:
        angle_axis (ndarray): 3-vector angle-axis
        pt (ndarray): (3, N) points

    Returns:
        result (ndarray): rotated points
    """
    aa = angle_axis[:3].reshape((1,3))
    theta2 = aa.dot(aa.T)[0,0]
    if theta2 > 0:
        theta = np.sqrt(theta2)
        w = aa / theta
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        w_cross_pt = cross_product_mat(w[0]).dot(pt)

        w_dot_pt = w.dot(pt)

        result = pt * cos_theta + w_cross_pt * sin_theta + (w.T * (1 - cos_theta)).dot(w_dot_pt)
    else:
        w_cross_pt = cross_product_mat(aa[0,:]).dot(pt)
        result = pt + w_cross_pt
    return result


def reprojection_error_mot_str(match_idx, match_points, f, px, py, motion, structure):
    """
    Computes the reprojection error for all observed points.

    Args:
        match_idx (ndarray): MxN indices of visible points per camera
        match_points (ndarray): observed 2D image coordinates
        f (float): focal length
        px (float): principal point x
        py (float): principal point y
        motion (ndarray): camera motion parameters (angle-axis + translation)
        structure (ndarray): estimated 3D structure

    Returns:
        errors (ndarray): 1D vector of reprojection errors
    """
    N = match_idx.shape[0]

    errors = None
    for i in range(N):
        valid_pts = match_idx[i,:] >= 0
        valid_idx = match_idx[i, valid_pts]

        RP = angle_axis_rotate(motion[i, :, 0], structure[valid_pts,:].T)

        TRX = RP[0, :] + motion[i, 0, 1]
        TRY = RP[1, :] + motion[i, 1, 1]
        TRZ = RP[2, :] + motion[i, 2, 1]

        TRXoZ = TRX / TRZ
        TRYoZ = TRY / TRZ
        
        x = f * TRXoZ + px
        y = f * TRYoZ + py

        ox = match_points[valid_idx, 0]
        oy = match_points[valid_idx, 1]

        if errors is None:
            errors = np.vstack((x-ox, y-oy))
        else:
            errors = np.hstack((errors, np.vstack((x-ox, y-oy))))

    return errors.flatten()


def reprojection_error_mot_str_opt(mot_str, match_idx, match_points, f, px, py):
    """
    Wrapper to compute reprojection error from flattened motion + structure vector.

    Args:
        mot_str (ndarray): flattened motion and structure vector
        match_idx (ndarray): index map of observed 2D points
        match_points (ndarray): 2D observed points
        f (float): focal length
        px (float): principal point x
        py (float): principal point y

    Returns:
        errors (ndarray): reprojection error vector
    """
    num_cams = match_idx.shape[0]
    cut = 3 * 2 * num_cams
    motion = mot_str[:cut].reshape((num_cams, 3, 2))
    structure = mot_str[cut:].reshape((-1, 3))
    return reprojection_error_mot_str(match_idx, match_points, f, px, py, motion, structure)


def bundle_adjustment(frame):
    """
    Performs bundle adjustment to refine camera motion and 3D structure.

    Args:
        frame (Frame): scene containing camera views and structure

    Returns:
        None; updates frame.motion and frame.structure in-place
    """
    num_cameras = frame.motion.shape[0]
    motion_angle_axis = np.zeros((num_cameras, 3, 2))
    for i in range(num_cameras):
        motion_angle_axis[i, :, 0] = rotation_matrix_to_angle_axis(
                frame.motion[i,:, :-1])
        motion_angle_axis[i, :, 1] = frame.motion[i, :, -1]

    px = 0
    py = 0
    
    errors = reprojection_error_mot_str(frame.match_idx, frame.match_points, frame.focal_length, px, py, motion_angle_axis, frame.structure)

    vec = least_squares(reprojection_error_mot_str_opt, np.hstack((motion_angle_axis.flatten(), frame.structure.flatten())),
        args=(frame.match_idx, frame.match_points, frame.focal_length, px, py), method='lm')

    cut = 3 * 2 * num_cameras

    opt_val = vec['x']
    frame.structure = opt_val[cut:].reshape((-1,3))
    motion_angle_axis = opt_val[:cut].reshape((-1, 3, 2))

    for i in range(num_cameras):
        frame.motion[i,:,:] = np.hstack((angle_axis_to_rotation_matrix(motion_angle_axis[i,:,0]), motion_angle_axis[i,:,1].reshape((3,1))))


def multiply_transformations(A, B):
    """
    Multiplies two SE(3) transforms [R|t].

    Args:
        A (ndarray): 3x4 matrix
        B (ndarray): 3x4 matrix

    Returns:
        M (ndarray): resulting 3x4 matrix
    """
    return np.hstack((A[:,:3].dot(B[:,:3]), (A[:,:3].dot(B[:,-1]) + A[:,-1]).reshape((3,-1))))


def inverse(x):
    """
    Computes the inverse of a 3x4 transformation matrix.

    Args:
        x (ndarray): 3x4 transformation matrix

    Returns:
        inv_x (ndarray): inverse 3x4 matrix
    """
    return np.hstack((x[:3, :3].T, -x[:3, :3].T.dot(x[:3, -1]).reshape((3,-1))))


def transform_points(points_3d, Rt, is_inverse=False):
    """
    Applies a transformation matrix to a set of 3D points.

    Args:
        points_3d (ndarray): Nx3 points
        Rt (ndarray): 3x4 transformation matrix
        is_inverse (bool): whether to invert the transform

    Returns:
        new_points (ndarray): Nx3 transformed points
    """
    if is_inverse:
        return Rt[:,:3].T.dot((points_3d - Rt[:,-1]).T).T
    return Rt[:,:3].dot(points_3d.T).T + Rt[:,-1]


def row_intersection(A, B):
    """
    Finds row-wise intersection of two matrices.

    Args:
        A (ndarray): matrix A
        B (ndarray): matrix B

    Returns:
        intersect (ndarray): rows in both A and B
        idA (ndarray): row indices in A
        idB (ndarray): row indices in B
    """
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
                   'formats':ncols * [A.dtype]}
    intersect = np.intersect1d(A.view(dtype), B.view(dtype))
    intersect = intersect.view(A.dtype).reshape((-1,ncols))
    idA = np.array([np.where(np.all(A==x, axis=1))[0][0] for x in intersect])
    idB = np.array([np.where(np.all(B==x, axis=1))[0][0] for x in intersect])
    return intersect, idA, idB


def row_set_diff(A, B):
    """
    Finds row-wise set difference of two matrices.

    Args:
        A (ndarray): matrix A
        B (ndarray): matrix B

    Returns:
        set_diff (ndarray): rows only in one of A or B
        idA (ndarray): indices of those rows in A
        idB (ndarray): indices of those rows in B
    """
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
                   'formats':ncols * [A.dtype]}
    set_diff = np.setdiff1d(A.view(dtype), B.view(dtype))
    set_diff = set_diff.view(A.dtype).reshape((-1,ncols))
    idA = []
    idB = []
    for x in set_diff:
        idx_in_A = np.where(np.all(A==x, axis=1))[0]
        idx_in_B = np.where(np.all(B==x, axis=1))[0]
        if len(idx_in_A) != 0:
            idA.append(idx_in_A[0])
        if len(idx_in_B) != 0:
            idB.append(idx_in_B[0])

    return set_diff, np.array(idA), np.array(idB)

def remove_outliers(frame, threshold=10.0):
    """
    Removes outlier 3D points based on reprojection error and viewing angle consistency.

    Args:
        frame (Frame): The frame containing structure and camera views.
        threshold (float): Threshold for squared pixel reprojection error.

    Returns:
        None. Modifies frame.structure and frame.match_idx in-place.
    """
    threshold *= threshold
    threshold_in_degree = 2.0
    threshold_in_cos = math.cos(float(threshold_in_degree) / 180 * math.pi)

    for i in range(frame.match_idx.shape[0]):
        X = frame.K.dot(transform_points(frame.structure, frame.motion[i,:,:]).T)
        xy = X[:2, :] / X[2, :]
        selector = np.where(frame.match_idx[i,:] >= 0)[0]
        diff = xy[:, selector].T - frame.match_points[frame.match_idx[i, selector],:]
        outliers = np.sum(diff**2, axis=1) > threshold
        
        pts2keep = np.array([True] * frame.structure.shape[0])
        pts2keep[selector[outliers]] = False

        frame.structure = frame.structure[pts2keep, :]
        frame.match_idx = frame.match_idx[:, pts2keep]

    # check viewing angle
    num_frames = frame.motion.shape[0]
    positions = np.zeros((3, num_frames))
    for i in range(num_frames):
        Rt = frame.motion[i, : , :]
        positions[:, i] = -Rt[:3, :3].T.dot(Rt[:,-1])
    
    view_dirs = np.zeros((3, frame.structure.shape[0], num_frames))
    for i in range(frame.match_idx.shape[0]-1):
        selector = np.where(frame.match_idx[i,:] >= 0)[0]
        camera_view_dirs = frame.structure[selector,:] - positions[:, i]
        dir_length = np.sqrt(np.sum(camera_view_dirs ** 2))
        camera_view_dirs = camera_view_dirs / dir_length
        view_dirs[:, selector, i] = camera_view_dirs.T

    for c1 in range(num_frames):
        for c2 in range(c1,num_frames):
            if c1 == c2: continue
            selector1 = np.where(frame.match_idx[c1,:] >= 0)[0]
            selector2 = np.where(frame.match_idx[c2,:] >= 0)[0]
            selector = np.array([x for x in selector1 if x in selector2])
            if len(selector) == 0:
                continue
            view_dirs_1 = view_dirs[:, selector, c1]
            view_dirs_2 = view_dirs[:, selector, c2]
            cos_angles = np.sum(view_dirs_1 * view_dirs_2, axis=0)
            outliers = cos_angles > threshold_in_cos

            pts2keep = np.array([True] * frame.structure.shape[0])
            pts2keep[selector[outliers]] = False
            frame.structure = frame.structure[pts2keep, :]
            frame.match_idx = frame.match_idx[:,pts2keep]

def merge_two_frames(frameA, frameB, length):
    """
    Merges two frames into a unified structure and motion representation.

    Args:
        frameA (Frame): First frame (already accumulated).
        frameB (Frame): New frame to merge in.
        length (int): Total number of cameras after merge.

    Returns:
        merged_frame (Frame): Unified frame object with merged structure and motion.
    """
    merged_frame = deepcopy(frameA)

    # Align frameB to frameA's coordinate frame
    transform = multiply_transformations(inverse(frameA.motion[-1]), frameB.motion[0])
    frameB.structure = transform_points(frameB.structure, transform)
    for i in range(2):
        frameB.motion[i] = multiply_transformations(frameB.motion[i], inverse(transform))

    # Add frameB's second camera to motion
    merged_frame.motion = np.vstack([merged_frame.motion, frameB.motion[-1][None]])

    # Reconcile matched 2D points
    trA = np.where(frameA.match_idx[0] >= 0)[0]
    trB = np.where(frameB.match_idx[0] >= 0)[0]
    xyA = frameA.match_points[frameA.match_idx[-1, trA]]
    xyB = frameB.match_points[frameB.match_idx[0, trB]]
    xy_common, iA, iB = row_intersection(xyA, xyB)

    merged_frame.match_idx = np.vstack([merged_frame.match_idx, neg_ones((1, merged_frame.match_idx.shape[1]))])
    for a_idx, b_idx in zip(iA, iB):
        idA = trA[a_idx]
        idB = trB[b_idx]
        pt_idx = frameB.match_idx[1, idB]
        merged_frame.match_points = np.vstack([merged_frame.match_points, frameB.match_points[pt_idx]])
        merged_frame.match_idx[length, idA] = merged_frame.match_points.shape[0] - 1

    # Add new points not yet seen in frameA
    xy_new, iB_new, _ = row_set_diff(xyB, xyA)
    for idx in iB_new:
        idB = trB[idx]
        pt_2d_0 = frameB.match_points[frameB.match_idx[0, idB]]
        pt_2d_1 = frameB.match_points[frameB.match_idx[1, idB]]
        pt_3d = frameB.structure[idB]

        merged_frame.match_points = np.vstack([merged_frame.match_points, pt_2d_0])
        merged_frame.match_idx = np.hstack([merged_frame.match_idx, neg_ones((merged_frame.match_idx.shape[0], 1))])
        merged_frame.match_idx[length - 1, -1] = merged_frame.match_points.shape[0] - 1

        merged_frame.structure = np.vstack([merged_frame.structure, pt_3d])
        merged_frame.match_points = np.vstack([merged_frame.match_points, pt_2d_1])
        merged_frame.match_idx[length, -1] = merged_frame.match_points.shape[0] - 1

    return merged_frame

def merge_all_frames(frames):
    """
    Iteratively merges a list of Frame objects into a complete scene.

    Args:
        frames (list of Frame): List of frames with relative motion and structure.

    Returns:
        merged_frame (Frame): Unified frame with consistent 3D structure and motion.
    """
    merged_frame = deepcopy(frames[0])
    for i in range(1, len(frames)):
        merged_frame = merge_two_frames(merged_frame, frames[i], i + 1)
        merged_frame.structure = triangulate(merged_frame)
        bundle_adjustment(merged_frame)
        remove_outliers(merged_frame, threshold=10.0)
        bundle_adjustment(merged_frame)

    return merged_frame
