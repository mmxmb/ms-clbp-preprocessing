from skimage.feature import local_binary_pattern, local_binary_pattern_magnitude
from skimage.transform import rescale
import cv2
import numpy as np
from typing import List, Tuple, NewType

Matrix = NewType('Matrix', np.ndarray)


def rescale_img(image: Matrix, scales: List[float]) -> List[Matrix]:
    """ Creates list of rescaled copies of `image`,
        each rescaled by corresponding factor from `scales`.
    """
    rescaled_imgs = []
    for scale in scales:
        if scale == 1:
            rescaled_imgs.append(np.copy(image))
        else:
            rescaled_imgs.append(rescale(image, scale))
    return rescaled_imgs


def overlapping_patch_partition(image: Matrix, patch_size: int) -> List[Matrix]:
    """ Partitions `image` into `patch_size` ☓ `patch_size` overlapped
        patches in `image` grid. Overlap between two patches is `patch_size / 2`.
    """
    assert(len(image.shape) == 2)
    h, w = image.shape
    img = image.copy()
    step = patch_size // 2
    n_vert_patch, n_horiz_patch = h // step, w // step
    if not (h % step == 0 and w % step == 0):
        img = crop_center(img, n_horiz_patch * step, n_vert_patch * step)
    patches = []
    for y_offset in range(0, h - patch_size + 1, step):
        for x_offset in range(0, w - patch_size + 1, step):
            patch = np.copy(img[y_offset:y_offset + patch_size,
                                x_offset:x_offset + patch_size])
            patches.append(patch)
    return patches


def crop_center(image: Matrix, crop_x: int, crop_y: int) -> Matrix:
    """ Crops center of `image`, such that the final shape is (`crop_y`, `crop_x`).
        https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    """
    y, x = image.shape
    start_x = x // 2 - (crop_x // 2)
    start_y = y // 2 - (crop_y // 2)
    return image[start_y:start_y + crop_y, start_x:start_x + crop_x].copy()


def variable_scale_ms_clbp(image: Matrix, scales: List[float], n_points: int, radius: int, patch_size: int, n_bins: int) -> Matrix:
    """ Calculate Multi-Scale CLBP for a given radius of LBP operator.
    """
    rescaled_imgs = rescale_img(image, scales)  # create rescaled copies of image
    feature_matrix = []
    for rescaled_img in rescaled_imgs:

        # calculate CLBP coded image (sign component)
        clbp_s = local_binary_pattern(rescaled_img, n_points, radius, 'uniform')

        # calculate CLBP coded image (magnitude component)
        clbp_m = local_binary_pattern_magnitude(rescaled_img, n_points, radius)

        # partition CLBP images into overlapping patches
        clbp_s_patches = overlapping_patch_partition(clbp_s, patch_size)
        clbp_m_patches = overlapping_patch_partition(clbp_m, patch_size)

        # calculate feature vector for each patch
        for clbp_s_patch, clbp_m_patch in zip(clbp_s_patches, clbp_m_patches):

            # calculate occurrence histogram for both patch components
            clbp_s_hist, _ = np.histogram(clbp_s_patch, range(n_bins + 1))
            clbp_m_hist, _ = np.histogram(clbp_m_patch, range(n_bins + 1))
            # concatenate two histograms to get histogram feature vector
            hist_feature_vector = np.append(clbp_s_hist, clbp_m_hist)

            # append histogram feature vector to feature matrix
            feature_matrix.append(hist_feature_vector)

    return np.array(feature_matrix)


def ms_clbp_feature_matrices(image: Matrix, scales: List[float], n_points: int, radii: List[int], patch_size: int, n_bins: int) -> List[Matrix]:
    """ Calculate Multi-Scale CLBP for a range of radii of LBP operator.
    """
    matrices = []
    for radius in radii:
        feature_matrix = variable_scale_ms_clbp(image, scales, n_points, radius, patch_size, n_bins)
        matrices.append(feature_matrix)
    return matrices
