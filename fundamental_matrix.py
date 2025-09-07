import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

def lls_eight_point_alg(points1, points2):
    """
    LLS_EIGHT_POINT_ALG computes the fundamental matrix from matching points using 
    the linear least squares eight-point algorithm.

    Args:
        points1 - (N, 3) array of homogeneous coordinates in image 1.
        points2 - (N, 3) array of homogeneous coordinates in image 2.

    Returns:
        F - (3, 3) fundamental matrix such that (points2)^T * F * points1 = 0.

    Hints:
        - Use np.kron or manual multiplication to construct matrix A.
        - Solve Af = 0 using SVD and reshape f into a 3x3 matrix.
        - Enforce rank-2 constraint on F by zeroing out the last singular value.
    """
    n = points1.shape[0]
    if points2.shape[0] != n:
        raise ValueError("Number of points must match in both images.")
    
    F = np.zeros((3, 3))
    ### START YOUR CODE ###
    # First we will build up the Matrix A, recall the formula is x'^T * F * x
    # To build up from the point correspondces means to multiple x'
    # and x
    x1 = points1[:, 0]
    y1 = points1[:, 1]
    z1 = points1[:, 2]

    x2 = points2[:, 0]
    y2 = points2[:, 1]
    z2 = points2[:, 2]

    A = np.column_stack([
      x2*x1, x2*y1, x2*z1,
      y2*x1, y2*y1, y2*z1,
      z2*x1, z2*y1, z2*z1
    ])

    # "Solve Af = 0 using SVD and reshape f into a 3x3 matrix"
    u_a, s_a, v_a_t = np.linalg.svd(A)
    F_reshape = (v_a_t[-1,:]).reshape(3, 3)

    # Enforce rank-2 constraint on F by zeroing out the last singular value.
    # Obviously to get the singular value, we need to get svd
    u_f, s_f, v_f_t = np.linalg.svd(F_reshape)
    F_zerosig = np.diag([s_f[0], s_f[1], 0.0])
    # Now rebuild up F
    F = u_f@ F_zerosig @ v_f_t


    ### END YOUR CODE ###
    return F


def normalized_eight_point_alg(points1, points2):
    """
    NORMALIZED_EIGHT_POINT_ALG computes the fundamental matrix using the normalized
    eight-point algorithm.

    Args:
        points1 - (N, 3) array of homogeneous coordinates in image 1.
        points2 - (N, 3) array of homogeneous coordinates in image 2.

    Returns:
        F - (3, 3) fundamental matrix such that (points2)^T * F * points1 = 0.

    Hints:
        - Normalize both sets of points so that their centroid is at the origin and average distance is sqrt(2).
        - Use LLS method on normalized points.
        - Denormalize the resulting F: F = T2.T @ F_normalized @ T1
    """
    F = np.zeros((3, 3))
    ### START YOUR CODE ###
    # Dehomogenize our coordinates to normalize them
    x1 = points1[:,0] / points1[:,2]
    y1 = points1[:,1] / points1[:,2]
    x2 = points2[:,0] / points2[:,2]
    y2 = points2[:,1] / points2[:,2]
    

    # Normalize both sets of points so that their centroid is at the origin and average distance is sqrt(2).
    meanx1, meany1 = np.mean(x1), np.mean(y1)
    meanx2, meany2 = np.mean(x2), np.mean(y2)
    x1_centered = x1-meanx1
    y1_centered = y1-meany1
    x2_centered = x2-meanx2
    y2_centered = y2-meany2

    # Now we compute the rms distance, NOT THE EUCLIDEAN distance. Idk how on earth
    # we were supposed to know this, but learned from oh. Never covered in lecture...
    dist1 = np.sqrt(np.mean(x1_centered**2+y1_centered**2))
    dist2 = np.sqrt(np.mean(x2_centered**2+y2_centered**2))
    scale_1 = np.sqrt(2)/dist1
    scale_2 = np.sqrt(2) /dist2

    # Build up transformation matrix from lecture to both translate and scale
    T_1 = np.array([[scale_1, 0, -scale_1 * meanx1],[0, scale_1, -scale_1 * meany1],
        [0, 0, 1]])

    T_2 = np.array([[scale_2, 0, -scale_2 * meanx2],[0, scale_2, -scale_2 * meany2],
        [0, 0, 1]])
    normalized_1 = (T_1 @ points1.T).T
    normalized_2 = (T_2 @ points2.T).T
    F_normalized = lls_eight_point_alg(normalized_1, normalized_2)
    
    # "Denormalize the resulting F: F = T2.T @ F_normalized @ T1"
    F = T_2.T @ F_normalized @ T_1

    ### END YOUR CODE ###
    return F


def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    """
    PLOT_EPIPOLAR_LINES_ON_IMAGES visualizes epipolar lines for corresponding points
    on a pair of images.

    Args:
        points1 - (N, 3) array of homogeneous points in image 1.
        points2 - (N, 3) array of homogeneous points in image 2.
        im1 - First image (HxW or HxWxC).
        im2 - Second image.
        F - (3, 3) fundamental matrix.

    Returns:
        None (displays a matplotlib figure with epipolar lines).
    """
    def plot_lines(img, pts_src, pts_dst, Fmat):
        lines = Fmat.T @ pts_dst.T
        plt.imshow(img, cmap='gray')
        for i in range(lines.shape[1]):
            a, b, c = lines[:, i]
            x = np.array([0, img.shape[1]])
            y = -(a * x + c) / b
            plt.plot(x, y, 'r')
            plt.plot(pts_src[i, 0], pts_src[i, 1], 'b*')
        plt.axis([0, img.shape[1], img.shape[0], 0])

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plot_lines(im1, points1, points2, F)
    plt.axis('off')

    plt.subplot(122)
    plot_lines(im2, points2, points1, F.T)
    plt.axis('off')
    plt.show()


def compute_distance_to_epipolar_lines(points1, points2, F):
    """
    COMPUTE_DISTANCE_TO_EPIPOLAR_LINES computes the average distance of points in
    image 1 to the epipolar lines induced by points in image 2.

    Args:
        points1 - (N, 3) array of homogeneous coordinates in image 1.
        points2 - (N, 3) array of homogeneous coordinates in image 2.
        F - (3, 3) fundamental matrix such that points2.T * F * points1 = 0.

    Returns:
        average_distance - Scalar, the mean distance from points1 to their epipolar lines.
    """
    average_distance = 0.0

    ### START YOUR CODE ###
    total_dist = 0.0

    for i in range(points1.shape[0]):
      # If a, b, and c are the coefficents of the line, then we just multply F by x
      a, b, c = F.T @ points2[i]

      x, y = points1[i, 0] / points1[i, 2], points1[i, 1] / points1[i, 2]
      # Now just use the formula
      total_dist += (abs(a*x + b*y + c)) / (np.sqrt(a**2+b**2))

    average_distance = total_dist / points1.shape[0]
    ### END YOUR CODE ###

    return average_distance
