# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:18:39 2019

@author: yifei
"""
import community
import numpy as np
import scipy.stats as ss
import tree_structure_clustering
import scipy.spatial.distance as dist

def KL_distance(raw_p, raw_q, max_value):
    '''
    calculate KL divergence bewteen dataset p and q
    '''
    # re-orgnize the raw data
    m_bins = [x for x in range(0, max_value+2, 2)]
    # to prevent 0 items, manully add 0.1
    freq_p = np.histogram(raw_p, bins=m_bins)[0] + 0.1
    freq_q = np.histogram(raw_q, bins=m_bins)[0] + 0.1
    freq_p = freq_p / np.sum(freq_p)
    freq_q = freq_q / np.sum(freq_q)
    
    similarity = np.exp(-np.mean([freq_p * np.log(freq_p / freq_q), freq_q * np.log(freq_q / freq_p)]))
    
    return similarity
    
def build_data_Static_S(CCRF_Y):
    '''
    return ccrf_s
    
    CCRF_Y: shape(n, t)
    '''
    n, t = CCRF_Y.shape
    CCRF_S = np.zeros([n, n])
    I = np.eye(n)
    
    for i in range(0, n):
        for j in range(i, n):
            tmp_max = np.max([np.max(CCRF_Y[i, :]), np.max(CCRF_Y[j, :])]).astype(int)
            simi = KL_distance(CCRF_Y[i, :], CCRF_Y[j, :], tmp_max)
            CCRF_S[i, j] = simi
            CCRF_S[j, i] = CCRF_S[i, j]
    # make sure that S_{i, j} = 0
    CCRF_S = CCRF_S - I
    
    return CCRF_S

def check_entropy(x):
    
    x_len = x.shape[0]
    x_value_list = set([x[i] for i in range(x_len)])
    ent = 0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x_len
        log_p = np.log2(p)
        ent = ent + log_p
    
    return -ent

def clique_entropy(m_cliques, data):
    
    clique_len = len(m_cliques)
    ents = np.zeros([clique_len, 1])
    for i in range(0, clique_len):
        data_x = data[m_cliques[i], :].flatten()
        ents[i] = check_entropy(data_x)
    
    return np.mean(ents)

def cliques2dict(m_cliques):
    
    m_dict = {}
    for i in range(0, len(m_cliques)):
        for j in range(0, len(m_cliques[i])):
            m_dict.update({m_cliques[i][j]:i})
    return m_dict

def clique_modularity(m_cliques, similarity, var_weight):
    
    n = similarity.shape[0]
    threshold = np.mean(similarity) + var_weight * np.var(similarity)
    m_graph = tree_structure_clustering.buildGraph(similarity, n, threshold)
    m_partition = cliques2dict(m_cliques)
    m_modularity = community.modularity(m_partition, m_graph)
    
    return m_modularity, m_graph

def graph2matrix(m_graph):
    
    num_regions = m_graph.number_of_nodes()
    matrix = np.zeros([num_regions, num_regions])
    for edge in m_graph.edges():
        matrix[edge[0], edge[1]] = 1
        
    num_zeros = len(np.where(matrix == 0)[0])
    zero_ratio = num_zeros / num_regions**2
    return matrix, zero_ratio

def fully2partial(CCRF_S, m_graph):
    
    partial_matrix, zero_ratio = graph2matrix(m_graph)
    t = CCRF_S.shape[0]
    n = CCRF_S.shape[1]
    new_CCRF_S = np.zeros([t, n, n])
    for i in range(0, t):
        new_CCRF_S[i] = CCRF_S[i] * partial_matrix
    
    return new_CCRF_S, zero_ratio
    
def transfer_static_S(similarity, t):
    
    n = similarity.shape[0]
    new_CCRF_S = np.zeros([t, n, n])
    for i in range(0, t):
        new_CCRF_S[i] = similarity
    return new_CCRF_S