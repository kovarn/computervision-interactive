import cv2

import numpy as np

Image = np.ndarray
GrayScaleImage = np.ndarray
BinaryImage = np.ndarray


def kuwahara_filter(src: GrayScaleImage, kernel_size: int, convert: bool = True) -> GrayScaleImage:
    if src.dtype == np.uint8:
        print('Converting uint8 image to float64')
        src = src / 255.0
    elif np.max(src) > 1 or np.min(src) < 0:
        raise ValueError("Float type image must have values in range [0,1]")

    src_sq = np.square(src)
    window_size = int((kernel_size + 1) / 2)
    positions = ['tl', 'tr', 'br', 'bl']

    kernels_anchors_list = [getKernel(position, window_size) for position in positions]
    means_list = [cv2.filter2D(src, -1, kernel, anchor=anchor) for (kernel, anchor) in kernels_anchors_list]
    variances_list = [cv2.filter2D(src_sq, -1, kernel, anchor=anchor) - np.square(mean) for ((kernel, anchor), mean) in
                      zip(kernels_anchors_list, means_list)]
    min_var_indices = np.argmin(variances_list, axis=0)
    kuwahara = np.choose(min_var_indices, means_list)
    if convert:
        # return uint8 image
        return (kuwahara * 255).astype(np.uint8)
    else:
        # return float image (values in interval [0,1])
        return kuwahara


def getKernel(position, window_size):
    """
    Returns an averaging filter kernel of the specified size and the anchor position.
    In this implementation the kernel does not depend on position.
    :param position: position of the window.
    :param window_size: size of the returned kernel.
    :return: tuple containing kernel and anchor
    """
    if position == 'tl':
        anchor = (window_size - 1, window_size - 1)
    elif position == 'tr':
        anchor = (window_size - 1, 0)
    elif position == 'br':
        anchor = (0, 0)
    elif position == 'bl':
        anchor = (0, window_size - 1)
    else:
        raise IndexError("position should be one of 'tl', 'tr', 'br', 'bl'.")
    kernel = np.ones((window_size, window_size), dtype=np.float64) / (window_size ** 2)
    return kernel, anchor
