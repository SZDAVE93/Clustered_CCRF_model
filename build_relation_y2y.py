# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:54:50 2018

@author: szdave
"""

import numpy as np

def KL_distance(raw_p, raw_q):
    '''
    calculate KL divergence bewteen dataset p and q
    '''
    # re-orgnize the raw data
    m_bins = [x for x in range(0, 101, 5)]
    # to prevent 0 items, manully add 0.1
    freq_p = np.histogram(raw_p, bins=m_bins)[0] + 0.1
    freq_q = np.histogram(raw_q, bins=m_bins)[0] + 0.1
    freq_p = freq_p / np.sum(freq_p)
    freq_q = freq_q / np.sum(freq_q)
    
    similarity = np.exp(-np.mean([freq_p * np.log(freq_p / freq_q), freq_q * np.log(freq_q / freq_p)]))
    
    return similarity
    
def build_data_S(CCRF_Y):
    '''
    return ccrf_s
    
    CCRF_Y: shape(n, t)
    '''
    n, t = CCRF_Y.shape
    CCRF_S = np.zeros([t, n, n])
    I = np.eye(n)
    
    for k in range(t-1, t):
        for i in range(0, n):
            for j in range(i, n):
                temp = KL_distance(CCRF_Y[i, 0:k], CCRF_Y[j, 0:k])
                CCRF_S[k, i, j] = temp
                CCRF_S[k, j, i] = CCRF_S[k, i, j]
        # make sure that S_{i, j} = 0
        CCRF_S[k] = CCRF_S[k] - I
    
    return CCRF_S
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
