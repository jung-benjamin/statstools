#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Functions to calculate statistical tolerance intervals

Currently three types of tolerance intervals can be calculated
with this moduel: tolerance interval for normally distributed
data, non-parametric tolerance intervals and content corrected
tolerance intervall.

@author jung
"""

import numpy as np
from scipy.stats import norm, binom, chi2
from statsmodels.distributions.empirical_distribution import ECDF

def k_factor(dof, N, p, conf):
    """Calculate the k-factor for tolerance limits
    
    The k-factor is valid for tolerance limits for 
    samples drawn from a normal distribution.
    """
    prop_inv = (1.0 - p) / 2.0
    gauss_critical = norm.isf(prop_inv)
    chi_critical = chi2.isf(q=conf, df=dof)
    k = np.sqrt((dof * (1 + (1/N)) * gauss_critical**2) / chi_critical)
    return k

def tolerance_interval(data, prop, conf, axis=0):
    """Calculate a tolerance interval
    
    Approximate a tolerance interval that contains 
    a portion prop of the data with a confidence
    level conf, assuming the data is normally 
    distributed.
    
    Parameters
    ----------
    data     sampled data, numpy.ndarray
    prop     portion of distribution to be 
             inside the interval as a fraction, float
    conf     confidence level of the interval
    axis     axis of data array to estimate statistics
    
    Output
    ------
    bounds   upper and lower bound of interval,
             list
    """
    N = data.shape[axis]
    dof = N - 1
    prop_inv = (1.0 - prop) / 2.0
    gauss_critical = norm.isf(prop_inv)
    chi_critical = chi2.isf(q=conf, df=dof)
    interval = np.sqrt((dof * (1 + (1/N)) * gauss_critical**2) / chi_critical)
    upper = data.mean(axis = axis) + interval * data.std(ddof = 1, axis = axis)
    lower = data.mean(axis = axis) - interval * data.std(ddof = 1, axis = axis)
    return np.array([lower, upper])

def non_parametric_tolerance_interval(data, p, axis=0):
    """Calculate a tolerance interval for a non parametric distribution.
    
    Derivation and explanation is found in Statistical Tolerance
    Regions. The interval uses the highest and lowest values
    of the sample, i.e. the highest and lowest order statistic
    as bounds and calculates the confidence level that a 
    certaint portion of the distribution is in that interval.
    """
    n = data.shape[axis]
    alpha = (1 - n) * p**n + n * p**(n - 1)
    lower = data.min(axis = axis)
    upper = data.max(axis = axis)
    return {'Conf level': 1 - alpha, 'Bounds': np.array([lower, upper])}

def non_parametric_tolerance_interval_m(data, m, p, axis=0):
    """Calculate a tolerance interval for a non parametric distribution.
    
    Derivation and explanation is found in Statistical Tolerance
    Regions. The interval uses the m-th order statistic
    as bounds and calculates the confidence level that a 
    certaint portion of the distribution is in that interval.
    """
    n = data.shape[axis]
    if not isinstance(m, int):
        print('m should be an integer')
    if m%2 == 0:
        low = int(m / 2)
        upp = int(n - m / 2)
    else:
        low = int((m + 1) / 2)
        upp = int(n  + 1 - (m + 1) / 2)
    order_statistics = np.sort(data, axis = axis)
    q = n - m - 1
    conf = binom.cdf(q, n, p)
    
    lower = np.take(order_statistics, low, axis = axis)
    upper = np.take(order_statistics, upp, axis = axis)
    
    return {'Conf level': conf, 'Bounds': np.array([lower, upper])}

def ecdf(data, x):
    """Determine the empirical distribution function
    
    Calculate the empirical distribution function for a
    data sample and evaluate at a given point.
    """
    n = len(data)
    num = len(np.where(data < x)[0])
    return num / n

def D_n(data, bootstrap, k):
    """Quantity for content correcting tolerance intervals
    
    The quantity is made up of empirical distribution functions
    of the data sample and a bootstrap sample. The ecdfs are 
    evaluated at the upper and lower bounds of the non-corrected
    tolerance intervals for the bootstrap sample.
    """
    if len(data) != len(bootstrap):
        print('Sample and bootstrap sample do not have the same length.')
        return 0
    X_star = bootstrap.mean()
    S_star = bootstrap.std(ddof = 1)
    n = len(data)
    upper = X_star + k * S_star
    lower = X_star - k * S_star
    D_n = np.sqrt(n) * (
          ecdf(bootstrap, upper) - ecdf(bootstrap, lower)
          - (ecdf(data, upper) - ecdf(data, lower))
          )
    return D_n

def D_n_ecdf(data, bootstrap, k):
    """Quantity for content correcting tolerance intervals
    
    The quantity is made up of empirical distribution functions
    of the data sample and a bootstrap sample. The ecdfs are 
    evaluated at the upper and lower bounds of the non-corrected
    tolerance intervals for the bootstrap sample.
    
    Uses the ECDF function of the statsmodel module
    """
    if len(data) != len(bootstrap):
        print('Sample and bootstrap sample do not have the same length.')
        return 0
    X_star = bootstrap.mean()
    S_star = bootstrap.std(ddof = 1)
    n = len(data)
    upper = X_star + k * S_star
    lower = X_star - k * S_star
    ecdf = ECDF(data)
    ecdf_star = ECDF(bootstrap)
    D_n = np.sqrt(n) * (
          ecdf_star(upper) - ecdf_star(lower)
          - (ecdf(upper) - ecdf(lower))
          )
    return D_n

def p_star(data, k, d_star):
    """Calculate corrected content of tolerance interval"""
    X = data.mean()
    S = data.std(ddof = 1)
    n = len(data)
    p_star = ecdf(data, X + k * S) - ecdf(data, X - k * S) - d_star / np.sqrt(n)
    return p_star

def content_correction(data, p=0.95, conf=0.99, no_B=1999):
    """Calculate a corrected portion via bootstrapping
    
    The portion p that lies within a tolerance interval with
    confidence conf is corrected with bootstrapping methods.
    The tolerance limits with content p are computed by 
    assuming a normal distribution, and the corrected content 
    describes how much of the actual distribution lies within 
    those borders.
    
    Source: Content-Corrcted Tolerance Limits Based on the 
            Bootstrap, 
            https://doi.org/10.1198/004017001750386260 
    
    Parameters
    ----------
    data    np.ndarray, the original data
    conf    float %/100, confidence level
            of the calculated tolerance interval
    no_B    int, number of bootstrap resamples
    
    Output
    ------
    corrected_portion   corrected portion of content that
                        lies in the 
    """
    k = k_factor(dof = data.shape[0] - 1,
                 N = data.shape[0],
                 p = p,
                 conf = conf
                )
    
    D_n_boot = []
    resample = np.random.randint(len(data), size = (no_B, len(data)))
    for B in resample:
        D_n_star = D_n_ecdf(data = data, 
                       bootstrap = data[B], 
                       k = k)
        D_n_boot.append(D_n_star)

    integer_m = int((no_B + 1) * conf)

    D_n_sorted = np.sort(D_n_boot)
    d_gamma_star = D_n_sorted[integer_m]
    corrected_portion = p_star(data, k, d_gamma_star)
    
    return corrected_portion
