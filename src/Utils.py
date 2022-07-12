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
    toRet = np.zeros((num_classes, labelsVec.shape[1]))
    for i in range(labelsVec.shape[1]):
        toRet[labelsVec[0, i], i] = 1
    return toRet


def getWindows(input: np.ndarray, kernel_size: int, padding: int = 0, stride: int = 1, dilate: bool = False):
    working_input = input
    working_pad = padding
    # dilate the input if necessary
    if dilate:
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
        working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

    # pad the input if necessary
    if working_pad != 0:
        working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)),
                               mode='constant',
                               constant_values=(0.,))

    batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides
    return np.lib.stride_tricks.as_strided(
        working_input,
        (working_input.shape[0], working_input.shape[1], working_input.shape[2] - kernel_size + 1, working_input.shape[3] - kernel_size + 1,
         kernel_size, kernel_size),
        (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
    )
