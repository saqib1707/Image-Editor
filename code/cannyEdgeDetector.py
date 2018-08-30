# from CannyEdge.utils import to_ndarray
# from CannyEdge.core import (gs_filter, gradient_intensity, suppression,
#                             threshold, tracking)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import copy
import argparse
from scipy import misc
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage

# Argparse
parser = argparse.ArgumentParser(description='Educational Canny Edge Detector')
parser.add_argument('source', metavar='src', help='image source (jpg, png)')
parser.add_argument('sigma', type=float, metavar='sigma', help='Gaussian smoothing parameter')
parser.add_argument('t', type=int, metavar='t', help='lower threshold')
parser.add_argument('T', type=int, metavar='T', help='upper threshold')
parser.add_argument("--all", help="Plot all in-between steps")
args = parser.parse_args()

def to_ndarray(img):
    im = misc.imread(img, flatten=True)
    # cv2.imshow('test', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    im = im.astype('int32')
    return im

def gs_filter(img, sigma):
    """ Step 1: Gaussian filter

    Args:
        img: Numpy ndarray of image
        sigma: Smoothing parameter

    Returns:
        Numpy ndarray of smoothed image
    """
    if type(img) != np.ndarray:
        raise TypeError('Input image must be of type ndarray.')
    else:
        return gaussian_filter(img, sigma)

def gradient_intensity(img):
    """ Step 2: Find gradients

    Args:
        img: Numpy ndarray of image to be processed (denoised image)

    Returns:
        G: gradient-intensed image
        D: gradient directions
    """

    # Kernel for Gradient in x-direction
    Kx = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
    )
    # Kernel for Gradient in y-direction
    Ky = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
    )
    # Apply kernels to the image
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    # return the hypothenuse of (Ix, Iy)
    G = np.hypot(Ix, Iy)
    D = np.arctan2(Iy, Ix)
    return (G, D)

def suppression(img, D):
    """ Step 3: Non-maximum suppression

    Args:
        img: Numpy ndarray of image to be processed (gradient-intensed image)
        D: Numpy ndarray of gradient directions for each pixel in img

    Returns:
        ...
    """

    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)

    for i in range(M):
        for j in range(N):
            # find neighbour pixels to visit from the gradient directions
            where = round_angle(D[i, j])
            try:
                if where == 0:
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                        Z[i,j] = img[i,j]
                elif where == 90:
                    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                        Z[i,j] = img[i,j]
                elif where == 135:
                    if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
                        Z[i,j] = img[i,j]
                elif where == 45:
                    if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
                        Z[i,j] = img[i,j]
            except IndexError as e:
                """ Todo: Deal with pixels at the image boundaries. """
                pass
    import pdb;pdb.set_trace()
    return Z


def threshold(img, t, T):
    """ Step 4: Thresholding
    Iterates through image pixels and marks them as WEAK and STRONG edge
    pixels based on the threshold values.

    Args:
        img: Numpy ndarray of image to be processed (suppressed image)
        t: lower threshold
        T: upper threshold

    Return:
        img: Thresholdes image

    """
    # define gray value of a WEAK and a STRONG pixel
    cf = {
        'WEAK': np.int32(50),
        'STRONG': np.int32(255),
    }

    # get strong pixel indices
    strong_i, strong_j = np.where(img > T)

    # get weak pixel indices
    weak_i, weak_j = np.where((img >= t) & (img <= T))

    # get pixel indices set to be zero
    zero_i, zero_j = np.where(img < t)

    # set values
    img[strong_i, strong_j] = cf.get('STRONG')
    img[weak_i, weak_j] = cf.get('WEAK')
    img[zero_i, zero_j] = np.int32(0)

    return (img, cf.get('WEAK'))

def tracking(img, weak, strong=255):
    """ Step 5:
    Checks if edges marked as weak are connected to strong edges.

    Note that there are better methods (blob analysis) to do this,
    but they are more difficult to understand. This just checks neighbour
    edges.

    Also note that for perfomance reasons you wouldn't do this kind of tracking
    in a seperate loop, you would do it in the loop of the tresholding process.
    Since this is an **educational** implementation ment to generate plots
    to help people understand the major steps of the Canny Edge algorithm,
    we exceptionally don't care about perfomance here.

    Args:
        img: Numpy ndarray of image to be processed (thresholded image)
        weak: Value that was used to mark a weak edge in Step 4

    Returns:
        final Canny Edge image.
    """

    M, N = img.shape
    for i in range(M):
        for j in range(N):
            if img[i, j] == weak:
                # check if one of the neighbours is strong (=255 by default)
                try:
                    if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
                         or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
                         or (img[i+1, j + 1] == strong) or (img[i-1, j - 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def round_angle(angle):
    """ Input angle must be in [0,180) """
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif (22.5 <= angle < 67.5):
        angle = 45
    elif (67.5 <= angle < 112.5):
        angle = 90
    elif (112.5 <= angle < 157.5):
        angle = 135
    return angle

def ced(img_file, sigma, t, T, all=False):
    img = to_ndarray(img_file)
    if not all:
        # avoid copies, just do all steps:
        img = gs_filter(img, sigma)
        img, D = gradient_intensity(img)
        img = suppression(img, D)
        img, weak = threshold(img, t, T)
        img = tracking(img, weak)
        return [img]
    else:
        # make copies, step by step
        img1 = gs_filter(img, sigma)
        img2, D = gradient_intensity(img1)
        img3 = suppression(copy(img2), D)
        img4, weak = threshold(copy(img3), t, T)
        img5 = tracking(copy(img4), weak)
        return [to_ndarray(img_file), img1, img2, img3, img4, img5]

def plot(img_list, safe=False):
    for d, img in enumerate(img_list):
        plt.subplot(1, len(img_list), d+1), plt.imshow(img, cmap='gray'),
        plt.xticks([]), plt.yticks([])
    plt.show()

img_list = ced(args.source, args.sigma, args.t, args.T, all=args.all)
plot(img_list)
