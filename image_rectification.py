import numpy as np
from fundamental_matrix import *

def compute_epipole(points1, points2, F):
    """
    COMPUTE_EPIPOLE computes the epipole (right nullspace of F^T) in homogeneous coordinates.

    Args:
        points1 - (N, 3) array of homogeneous points in the first image.
        points2 - (N, 3) array of homogeneous points in the second image.
        F - (3, 3) fundamental matrix such that points2^T * F * points1 = 0.

    Returns:
        epipole - (3,) homogeneous coordinates [x, y, 1] of the epipole in the second image.

    Hints:
        - The epipole is the right null vector of F^T * points2.T.
    """
    ### START YOUR CODE ###
    # To do this, we can use svd to get the right singular vectors, and then get the vectors
    # that correspond to the eigenvalues of 0
    u, s, v_t = np.linalg.svd(F)

    # Get the last singular vector with an negative index
    last_vector = u[:,-1]
    # Normalize the epipole
    return last_vector/last_vector[2]

    ### END YOUR CODE ###
    

def compute_matching_homographies(e2, F, im2, points1, points2):
    """
    COMPUTE_MATCHING_HOMOGRAPHIES returns homographies H1 and H2 that rectify a stereo pair.

    Args:
        e2 - (3,) homogeneous coordinates of the epipole in the second image.
        F - (3, 3) fundamental matrix between the two images.
        im2 - Image array of the second image.
        points1 - (N, 3) homogeneous points in the first image.
        points2 - (N, 3) homogeneous points in the second image.

    Returns:
        H1 - (3, 3) homography to rectify the first image.
        H2 - (3, 3) homography to rectify the second image.

    Hints:
        - Follow Hartley & Zisserman's rectification approach.
        - Translate image center to origin, rotate epipole to x-axis, then project to infinity.
        - Estimate affine transform H1 to align transformed points.
    """
    H1 = np.eye(3)
    H2 = np.eye(3)
    ### START YOUR CODE ###
    # Its easier to do H2 first, so we do this:
    # "Translate image center to origin, rotate epipole to x-axis, then project to infinity."
    # Then, for H1, we solve an linear least squares problem to find the affine transform
    # that aligns H1 and H2 with their points
    h, w = im2.shape[0], im2.shape[1]
    center_y = h / 2.0
    center_x = w / 2.0

    T = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]])
    # Now "translate iamge center to origin"
    translated = T @ e2
    e0 = translated[0]/translated[2]
    e1 = translated[1]/translated[2]

    # Now just calculate the values of the sides and the angle through pythagoream
    # theorem and arctan
    val = np.sqrt(e0**2+e1**2)
    theta = -np.arctan2(e1, e0)
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
    G = np.array([[1, 0, 0],[0, 1, 0],[-1 / val, 0, 1]])
    H2 = G @ R @ T

    points2_transformed = (H2 @ points2.T).T
    # Transforming like this gives us the y coordiantes to look for with the second image
    # Dehomogenize to construct the matrix
    x1 = points1[:,0] /points1[:,2]
    y1 = points1[:,1] / points1[:,2]

    A = np.column_stack([x1, y1, np.ones(len(x1))])
    # np.ones adds a column of 1 due to the affine transformation we wish to do here
    # And solve the least squares
    result, residuals, rank, sig = np.linalg.lstsq(A,points2_transformed[:,1] / points2_transformed[:,2])
    # As explained in the textbook, this will try to minmize the differnce between the transformed
    #version of the points in the first image vs the recitifed versions in the second
    d, e, f_val = result
    H1 = np.array([
        [1, 0, 0],
        [d, e, f_val],
        [0, 0, 1]
    ])

    ### END YOUR CODE ###

    return H1, H2
