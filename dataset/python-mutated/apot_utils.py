"""
This file contains utility functions to convert values
using APoT nonuniform quantization methods.
"""
import math
'Converts floating point input into APoT number\n    based on quantization levels\n'

def float_to_apot(x, levels, indices, alpha):
    if False:
        print('Hello World!')
    if x < -alpha:
        return -alpha
    elif x > alpha:
        return alpha
    levels_lst = list(levels)
    indices_lst = list(indices)
    min_delta = math.inf
    best_idx = 0
    for (level, idx) in zip(levels_lst, indices_lst):
        cur_delta = abs(level - x)
        if cur_delta < min_delta:
            min_delta = cur_delta
            best_idx = idx
    return best_idx
'Converts floating point input into\n    reduced precision floating point value\n    based on quantization levels\n'

def quant_dequant_util(x, levels, indices):
    if False:
        return 10
    levels_lst = list(levels)
    indices_lst = list(indices)
    min_delta = math.inf
    best_fp = 0.0
    for (level, idx) in zip(levels_lst, indices_lst):
        cur_delta = abs(level - x)
        if cur_delta < min_delta:
            min_delta = cur_delta
            best_fp = level
    return best_fp
'Converts APoT input into floating point number\nbased on quantization levels\n'

def apot_to_float(x_apot, levels, indices):
    if False:
        i = 10
        return i + 15
    idx = list(indices).index(x_apot)
    return levels[idx]