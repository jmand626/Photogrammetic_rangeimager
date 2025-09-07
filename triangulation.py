import numpy as np
import matplotlib.pyplot as plt

"""
ESTIMATE_INITIAL_RT computes 4 possible relative camera extrinsics [R|T] from an essential matrix.

Args:
    E - (3, 3) Essential matrix

Returns:
    RT - (4, 3, 4) Array of 3x4 camera extrinsics, each representing a possible [R|T] solution

Hints:
    - Use SVD on E.
    - Construct R from U, W, V^T.
    - The four configurations arise from two R and two T choices.
"""
def estimate_initial_RT(E):
    RT = np.zeros((4, 3, 4))
    ### START YOUR CODE ###
    u, s, v_t = np.linalg.svd(E)
    # We use a typical skew-symmetric matrix derived to preserve rotation and the desired
    # number of skewed values
    W = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])

    # Two valid rotation matrices
    R_1 = u @ W @ v_t
    R_2 = u @ W.T @ v_t

    # Two translationv ector from third column of u
    t_1 = u[:,2]
    t_2 = -t_1

    # Stack all combinations into 4 by 3 by 4 array
    RT[0] = np.hstack((R_1, t_1.reshape(3,1)))
    RT[1] = np.hstack((R_1,t_2.reshape(3,1)))
    RT[2] = np.hstack((R_2, t_1.reshape(3,1)))
    RT[3] = np.hstack((R_2, t_2.reshape(3,1)))


    ### END YOUR CODE ###
    return RT


"""
LINEAR_ESTIMATE_3D_POINT triangulates a 3D point from multiple views using the linear method.

Args:
    image_points - (M, 2) 2D points in each image
    camera_matrices - (M, 3, 4) camera projection matrices

Returns:
    point_3d - (3,) Estimated 3D point

Hints:
    - Construct the A matrix from the cross product constraint.
    - Solve Af = 0 via SVD.
"""
def linear_estimate_3d_point(image_points, camera_matrices):
    point_3d = np.zeros(3)
    ### START YOUR CODE ###
    num_views = len(image_points)
    A = np.zeros((num_views*2, 4))

    # Per each view
    for i in range(num_views):
      # Cross product constraint: we know x * Px = 0 produces two equations per camera
      # due to the two xs. Given that and a point (a,b),we know that b(Px) - Px = 0
      # and Px - a(Px) = 0 where we take differnt rows of P is true
      a, b = image_points[i]
      P = camera_matrices[i]

      A[2*i + 1] = P[0]- a * P[2]
      A[2*i] = b * P[2] - P[1]

    u, s, v_t = np.linalg.svd(A)
    x_unhom = v_t[-1] /v_t[-1][3]
    # To un homogenize
    point_3d = x_unhom[:3]
    ### END YOUR CODE ###
    return point_3d


"""
REPROJECTION_ERROR computes reprojection error of a 3D point given image measurements.

Args:
    point_3d - (3,) The estimated 3D point
    image_points - (M, 2) Measured 2D points in the images
    camera_matrices - (M, 3, 4) Camera projection matrices

Returns:
    error - (2M,) Reprojection error vector

Hints:
    - Project the 3D point into each view and compare with measured image point.
"""
def reprojection_error(point_3d, image_points, camera_matrices):
    error = np.zeros((2 * image_points.shape[0],))
    ### START YOUR CODE ###
    M = len(camera_matrices)
    X_hom = np.append(point_3d, 1)
    # Homogenize x

    for i in range(M):
        P = camera_matrices[i]
        a,b = image_points[i]
        # "Project the 3D point into each view"
        p_projected = P @ X_hom

        x_proj = p_projected[0]/p_projected[2]
        y_proj = p_projected[1]/p_projected[2]

        # Similar to the previous function, comparing projected with actual image points
        error[2*i] = x_proj-a
        error[2*i + 1] = y_proj-b
    ### END YOUR CODE ###
    return error


"""
JACOBIAN computes the Jacobian matrix of reprojection error w.r.t. the 3D point.

Args:
    point_3d - (3,) Estimated 3D point
    camera_matrices - (M, 3, 4) Projection matrices

Returns:
    jacobian - (2M, 3) Jacobian matrix
"""
def jacobian(point_3d, camera_matrices):
    jacobian = np.zeros((2 * camera_matrices.shape[0], 3))
    ### START YOUR CODE ###
    # Pretty similar to the last function as well
    M = len(camera_matrices)
    X_hom = np.append(point_3d, 1)
    # Homogenize x

    for i in range(M):
        P = camera_matrices[i]
        # "Project the 3D point into each view"
        p_projected = P @ X_hom
        p0, p1, p2 = p_projected

        # Our jacobians here are calculated through the power/quiotent rule
        jacobian[2*i] = [(P[0,0]*p2- p0*P[2,0])/p2**2,
            (P[0,1]*p2 - p0*P[2,1])/p2 ** 2,
            (P[0,2]*p2 -p0*P[2,2])/p2 ** 2
        ]
        jacobian[2*i+1] = [(P[1,0]*p2 - p1*P[2,0]) / p2**2,
            (P[1,1]*p2 - p1*P[2,1])/p2**2,
            (P[1,2]*p2 - p1*P[2,2]) / p2**2
        ]
        # del/del applied to (p_i/p2) = (delp_i/delX * p2 - p_i * del p2/del X)/p2^2
        
    ### END YOUR CODE ###
    return jacobian


"""
NONLINEAR_ESTIMATE_3D_POINT refines a 3D point estimate using iterative optimization.

Args:
    image_points - (M, 2) 2D image observations
    camera_matrices - (M, 3, 4) camera projection matrices

Returns:
    point_3d - (3,) Refined 3D point estimate

Hints:
    - Use Gauss-Newton update: P_t+1 = P_t - (JᵗJ)⁻¹ Jᵗ e
    - Initialize with linear triangulation
"""
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    point_3d = np.zeros(3)
    ### START YOUR CODE ###
    point_3d = linear_estimate_3d_point(image_points, camera_matrices)

    for i in range(10):
      # Compute reprojection error
      e = reprojection_error(point_3d, image_points, camera_matrices)
      # Compute Jacobian matrix
      J = jacobian(point_3d, camera_matrices)

      # Compute update by solving with the linalg function
      update = np.linalg.solve((J.T @ J), -(J.T @ e))

      point_3d += update
    ### END YOUR CODE ###
    return point_3d


"""
ESTIMATE_RT_FROM_E selects the correct [R|T] from four possibilities using the positive depth constraint.

Args:
    E - (3, 3) Essential matrix
    image_points - (N, 2, 2) Image point correspondences for N points
    K - (3, 3) Intrinsic matrix

Returns:
    correct_RT - (3, 4) The correct camera extrinsic matrix [R|T]
"""
def estimate_RT_from_E(E, image_points, K):
    correct_RT = np.zeros((3, 4))
    ### START YOUR CODE ###
    candidates_RT = estimate_initial_RT(E)
    # "Compute the four candidate [R|t] transformations using your estimate_initial_RT() function."

    max_count = float('-inf')

    # Create a projection matrix
    I_3 = np.eye(3)
    P1 = K @ np.hstack((I_3, np.zeros((3, 1))))
    for j in range(4):
      RT = candidates_RT[j]
      # Create projection matrix for the 2nd camera
      P2 = K @ RT
      curr = 0

      for i in range (image_points.shape[0]):
        # Construct camera matrices / get points
        point1 = image_points[i, 0]
        point2 = image_points[i, 1]

        # Triangulation function requires these points to be IN AN ARRAY!!!
        points_array = np.array([point1, point2])

        point_3d = nonlinear_estimate_3d_point(points_array, np.array([P1, P2]))
        point_3d = np.append(point_3d, 1)
        projected1 = P1 @ point_3d
        projected2 = P2 @ point_3d
        # Now actually apply our projection

        # And count posistive depth!!
        if projected1[2] > 0 and projected2[2] > 0:
            curr += 1


      if curr >max_count:
            correct_RT = RT            
            max_count = curr

    
    ### END YOUR CODE ###
    return correct_RT
