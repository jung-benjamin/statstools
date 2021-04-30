#! /usr/bin/env python3
"""Functions to optimize binning for histograms"""

import numpy as np
from math import ceil

def interquartile_range(data):
    """Calculate the interquartile range of a dataset"""
    q3, q1 = np.percentile(data, [75, 25])
    iqr = q3 -q1
    return iqr

def freedman_diaconis(data):
    """Calculate the bin width for a histogram."""
    n = len(data)
    iqr = interquartile_range(data)
    width = 2 * iqr / n**(1/3)
    return width

def bin_number(data):
    """Calculate the number of bins for a histogram"""
    width = freedman_diaconis(data)
    number = (np.max(data) - np.min(data)) / width
    return ceil(number)


