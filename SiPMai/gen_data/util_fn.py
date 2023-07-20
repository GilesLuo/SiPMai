import cv2
import numpy as np


def uniform_noise(img):
    mean, sigma = 10, 100
    a = 2 * mean - np.sqrt(12 * sigma)  # a = -14.64
    b = 2 * mean + np.sqrt(12 * sigma)  # b = 54.64
    noiseUniform = np.random.uniform(a, b, img.shape)
    imgUniformNoise = img + noiseUniform
    imgUniformNoise = np.uint8(cv2.normalize(imgUniformNoise, None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]
    imgUniformNoise = np.array(imgUniformNoise, dtype=np.uint8)
    return imgUniformNoise


def motion_blur(image, degree=12, angle=45):
    image = np.array(image)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def compress_binary_arr(arr):
    """Compress a binary tensor to int8 array"""
    flattened = arr.reshape(-1)
    int8_representation = np.packbits(flattened)
    return int8_representation


def decompress_binary_arr(npz_filename):
    # Loading compressed data and decompressing it
    data = np.load(npz_filename)
    arr_atom = data['arr_atom']
    arr_bound = data['arr_bond']
    adj_matrix = data['adj_matrix']
    molecule_points_height = data['molecule_points_height']

    arr_atom_shape = data['arr_atom_shape']
    arr_bond_shape = data['arr_bond_shape']

    # """Decompress an int8 array back to a binary tensor"""
    arr_atom = np.unpackbits(arr_atom)
    arr_atom = arr_atom.reshape(arr_atom_shape)

    arr_bound = np.unpackbits(arr_bound)
    arr_bound = arr_bound.reshape(arr_bond_shape)

    return arr_atom, arr_bound, adj_matrix, molecule_points_height
