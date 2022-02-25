#!/usr/bin/env python

from __future__ import print_function, division

from collections import OrderedDict
try:
    import pickle
except ImportError:
    import cPickle as pickle  # pylint: disable=import-error
from pkg_resources import parse_version

from PIL import Image
import numpy as np
assert parse_version(np.__version__) >= parse_version('1.9.0'), \
    "numpy >= 1.9.0 is required for daltonize"
try:
    import matplotlib as mpl
    _NO_MPL = False
except ImportError:
    _NO_MPL = True


def transform_colorspace(img, mat):
    """Transform image to a different color space.
    """
    return np.einsum("ij, ...j", mat, img)


def simulate(img, color_deficit="d"):
    """Simulate the effect of color blindness on an image.

    img : PIL.PngImagePlugin.PngImageFile, input image
    color_deficit : {"d", "p", "t"}, optional
        type of colorblindness, d for deuteronopia (default),
        p for protonapia,
        t for tritanopia
    """
    # Colorspace transformation matrices
    cb_matrices = {
        "d": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]),
        "p": np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]),
        "t": np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]]),
    }
    rgb2lms = np.array([[17.8824, 43.5161, 4.11935],
                        [3.45565, 27.1554, 3.86714],
                        [0.0299566, 0.184309, 1.46709]])
    # Precomputed inverse
    lms2rgb = np.array([[8.09444479e-02, -1.30504409e-01, 1.16721066e-01],
                        [-1.02485335e-02, 5.40193266e-02, -1.13614708e-01],
                        [-3.65296938e-04, -4.12161469e-03, 6.93511405e-01]])

    img = img.copy()
    img = img.convert('RGB')

    rgb = np.asarray(img, dtype=float)
    # first go from RBG to LMS space
    lms = transform_colorspace(rgb, rgb2lms)
    # Calculate image as seen by the color blind
    sim_lms = transform_colorspace(lms, cb_matrices[color_deficit])
    # Transform back to RBG
    sim_rgb = transform_colorspace(sim_lms, lms2rgb)
    return sim_rgb


def daltonize(rgb, color_deficit='d'):
    """
    Adjust color palette of an image to compensate color blindness.
    """
    sim_rgb = simulate(rgb, color_deficit)
    err2mod = np.array([[0, 0, 0], [0.7, 1, 0], [0.7, 0, 1]])
    rgb = rgb.convert('RGB')
    err = transform_colorspace(rgb - sim_rgb, err2mod)
    dtpn = err + rgb
    return dtpn


def array_to_img(arr):
    """Convert a numpy array to a PIL image.
    """
    # clip values to lie in the range [0, 255]
    arr = clip_array(arr)
    arr = arr.astype('uint8')
    img = Image.fromarray(arr, mode='RGB')
    return img


def clip_array(arr, min_value=0, max_value=255):
    """Ensure that all values in an array are between min and max values.
        clipped such that all values are min_value <= arr <= max_value
    """
    comp_arr = np.ones_like(arr)
    arr = np.maximum(comp_arr * min_value, arr)
    arr = np.minimum(comp_arr * max_value, arr)
    return arr

# def _prepare_for_transform(fig):
#     """
#     Gather color keys/info for mpl figure and arange them such that the image
#     simulate() or daltonize() routines can be called on them.
#     """
#     mpl_colors = get_mpl_colors(fig)
#     rgb, alpha = arrays_from_dict(mpl_colors)
#     return rgb, alpha, mpl_colors


# def _join_rgb_alpha(rgb, alpha):
#     """
#     Combine (m, n, 3) rgb and (m, n) alpha array into (m, n, 4) rgba.
#     """
#     rgb = clip_array(rgb, 0, 1)
#     r, g, b = np.split(rgb, 3, 2)  # pylint: disable=invalid-name, unbalanced-tuple-unpacking
#     rgba = np.concatenate((r, g, b, alpha.reshape(alpha.size, 1, 1)),
#                           axis=2).reshape(-1, 4)
#     return rgba


# def simulate_mpl(fig, color_deficit='d', copy=False):
#     """
#     Simulate color blindness on a matplotlib figure.

#     Arguments:
#     ----------
#     fig : matplotlib.figure.Figure
#     color_deficit : {"d", "p", "t"}, optional
#         type of colorblindness, d for deuteronopia (default),
#         p for protonapia,
#         t for tritanopia
#     copy : bool, optional
#         should simulation happen on a copy (True) or the original
#         (False, default)

#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#     """
#     if copy:
#         # mpl.transforms cannot be copy.deepcopy()ed. Thus we resort
#         # to pickling.
#         pfig = pickle.dumps(fig)
#         fig = pickle.loads(pfig)
#     rgb, alpha, mpl_colors = _prepare_for_transform(fig)
#     sim_rgb = simulate(array_to_img(rgb * 255), color_deficit) / 255
#     rgba = _join_rgb_alpha(sim_rgb, alpha)
#     set_mpl_colors(mpl_colors, rgba)
#     fig.canvas.draw()
#     return fig


# def daltonize_mpl(fig, color_deficit='d', copy=False):
#     """
#     Daltonize a matplotlib figure.

#     Arguments:
#     ----------
#     fig : matplotlib.figure.Figure
#     color_deficit : {"d", "p", "t"}, optional
#         type of colorblindness, d for deuteronopia (default),
#         p for protonapia,
#         t for tritanopia
#     copy : bool, optional
#         should daltonization happen on a copy (True) or the original
#         (False, default)

#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#     """
#     if copy:
#         # mpl.transforms cannot be copy.deepcopy()ed. Thus we resort
#         # to pickling.
#         pfig = pickle.dumps(fig)
#         fig = pickle.loads(pfig)
#     rgb, alpha, mpl_colors = _prepare_for_transform(fig)
#     dtpn = daltonize(array_to_img(rgb * 255), color_deficit) / 255
#     rgba = _join_rgb_alpha(dtpn, alpha)
#     set_mpl_colors(mpl_colors, rgba)
#     fig.canvas.draw()
#     return fig


if __name__ == '__main__':
    import argparse

    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=str)
    parser.add_argument("output_image", type=str)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", "--simulate", help="create simulated image",
                       action="store_true")
    group.add_argument("-d", "--daltonize",
                       help="adjust image color palette for color blindness",
                       action="store_true")
    parser.add_argument("-t", "--type", type=str, choices=["d", "p", "t"],
                        help="type of color blindness (deuteranopia, "
                        "protanopia, tritanopia), default is deuteranopia "
                        "(most common)")
    args = parser.parse_args()

    if args.simulate is False and args.daltonize is False:
        print("No action specified, assume daltonizing")
        args.daltonize = True
    if args.type is None:
        args.type = "d"

    orig_img = Image.open(args.input_image)

    if args.simulate:
        simul_rgb = simulate(orig_img, args.type)
        simul_img = array_to_img(simul_rgb)
        simul_img.save(args.output_image)
    if args.daltonize:
        dalton_rgb = daltonize(orig_img, args.type)
        dalton_img = array_to_img(dalton_rgb)
        dalton_img.save(args.output_image)
