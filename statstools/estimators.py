#! /usr/bin/env python3
"""Estimators for statistical parameters."""

import numpy as np


def robust_mode(x):
    """Calculate a robust estimator of the mode in 1 dimension.
    
    Source for the estimator: a post on stack-exchange. There are
    other sources given in the post, but I did not read them
    carefully.
    
    This implementation seems to work, but I have not tested it
    thoroughly.
    """
    order_statistic = np.array(sorted(x))
    if len(order_statistic) == 1:
        return order_statistic[0]
    elif len(order_statistic) == 2:
        return order_statistic.mean()
    elif len(order_statistic) == 3:
        if order_statistic[1] - order_statistic[0] > order_statistic[
                2] - order_statistic[1]:
            return order_statistic[[1, 2]].mean()
        else:
            return order_statistic[[0, 1]].mean()
    else:
        h1 = len(order_statistic) // 2
        k_vals, widths = [], []
        for k in range(len(order_statistic) - h1):
            width = order_statistic[k + h1] - order_statistic[k]
            k_vals.append(k)
            widths.append(width)
        best_k = k_vals[np.argmin(widths)]
        shortest = order_statistic[best_k:best_k + h1]
        return robust_mode(shortest)