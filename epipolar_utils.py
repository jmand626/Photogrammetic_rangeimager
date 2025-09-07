import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_data_from_txt_file(filename, use_subset=False):
    """
    Load 2D or 3D point data from a text file.

    Args:
        filename (str): Path to the data file.
        use_subset (bool): Unused flag; reserved for future extension.

    Returns:
        points (ndarray): Array of shape (N, 3). If the data is 2D, the third column is 1 (homogeneous).
                          If the data is 3D, the z-coordinate is read from the file.
    """
    with open(filename) as f:
        lines = f.read().splitlines()
    number_pts = int(lines[0])

    points = np.ones((number_pts, 3))
    for i in range(number_pts):
        split_arr = lines[i + 1].split()
        if len(split_arr) == 2:
            y, x = split_arr
        else:
            x, y, z = split_arr
            points[i, 2] = float(z)
        points[i, 0] = float(x)
        points[i, 1] = float(y)
    return points


def compute_rectified_image(im, H):
    """
    Apply a homography to rectify an image.

    Args:
        im (ndarray): Input grayscale image.
        H (ndarray): 3x3 homography matrix.

    Returns:
        new_image (ndarray): Warped image after applying the homography.
        offset (tuple): (min_x, min_y) offset used to keep image coordinates positive.

    Notes:
        This function manually computes the mapping from output pixels to original
        pixels using inverse warping. Nearest neighbor interpolation is used.
    """
    height, width = im.shape[:2]
    new_x = np.zeros((height, width))
    new_y = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            new_loc = H @ [x, y, 1]
            new_loc /= new_loc[2]
            new_x[y, x] = new_loc[0]
            new_y[y, x] = new_loc[1]

    offset_x = new_x.min()
    offset_y = new_y.min()
    new_x -= offset_x
    new_y -= offset_y
    new_height = int(np.ceil(new_y.max())) + 1
    new_width = int(np.ceil(new_x.max())) + 1

    H_inv = np.linalg.inv(H)
    new_image = np.zeros((new_height, new_width))

    for y in range(new_height):
        for x in range(new_width):
            orig_loc = H_inv @ [x + offset_x, y + offset_y, 1]
            orig_loc /= orig_loc[2]
            orig_x, orig_y = int(orig_loc[0]), int(orig_loc[1])

            if 0 <= orig_x < width and 0 <= orig_y < height:
                new_image[y, x] = im[orig_y, orig_x]

    return new_image, (offset_x, offset_y)


def scatter_3D_axis_equal(X, Y, Z, ax):
    """
    Plot 3D scatter points with equal scaling on all axes.

    Args:
        X (ndarray): 1D array of x coordinates.
        Y (ndarray): 1D array of y coordinates.
        Z (ndarray): 1D array of z coordinates.
        ax (Axes3D): Matplotlib 3D axis to plot on.

    Returns:
        None. This function modifies the provided axis.
    """
    ax.scatter(X, Y, Z)

    max_range = max(
        X.max() - X.min(),
        Y.max() - Y.min(),
        Z.max() - Z.min()
    ) / 2.0

    mid_x = (X.max() + X.min()) / 2.0
    mid_y = (Y.max() + Y.min()) / 2.0
    mid_z = (Z.max() + Z.min()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
