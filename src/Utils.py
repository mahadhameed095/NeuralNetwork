from math import floor
from time import time
import numpy as np
import timeit


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def oneHotEncode(labelsVec: np.ndarray, num_classes: int) -> np.ndarray:
    toRet = np.zeros((labelsVec.shape[0], num_classes))
    for i in range(labelsVec.shape[0]):
        toRet[i, labelsVec[i, 0]] = 1
    return toRet


def im2Col(input: np.ndarray, kernel_size: int, padding: int = 0, stride: int = 1):
    working_input = input
    batch_size, channels, features = working_input.shape
    dim = int(np.sqrt(features))
    out_size = floor((dim - kernel_size)/stride + 1)

    strides = (working_input.itemsize * features * channels,
               working_input.itemsize * features,
               working_input.itemsize * dim,
               working_input.itemsize)
    #dilate the input if necessary
    """
    if dilate:
        working_input = np.insert(working_input, range(1, working_input.shape[2]), 0, axis=2)
        working_input = np.insert(working_input, range(1, working_input.shape[3]), 0, axis=3)
    
    """
    #pad the input if necessary
    if padding != 0:
        working_input = working_input.reshape(working_input.shape[0], working_input.shape[1], dim, dim)
        working_input = np.pad(working_input, pad_width=((0,), (0,), (padding,), (padding,)),
                               mode='constant',
                               constant_values=(0.,))

    return (
        np.lib.stride_tricks.as_strided(
            working_input, 
            shape=(batch_size,channels, kernel_size, kernel_size, 1, 1, out_size, out_size), 
            strides=(strides[0], strides[1],         strides[2],         strides[3], 
                     strides[0], strides[1],stride * strides[2],stride * strides[3])
        )
        .reshape(batch_size, 
                 channels * kernel_size * kernel_size, 
                 out_size * out_size)
    )