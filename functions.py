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
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt

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

def read_results(select_type, time_window):
    
    
    step = 25
    results = np.zeros([20, 12])
    path = r'D:\yifei\Documents\Codes_on_GitHub\Clustered_CCRF_model\data\results\partial'
    data_rmse = np.load(r'{}\result_{}.npy'.format(path, 'rmse'))
    data_rank = np.load(r'{}\result_{}.npy'.format(path, 'rank'))
    '''
    step = 1
    results = np.zeros([20, 12])
    path = r'D:\yifei\Documents\Codes_on_GitHub\Clustered_CCRF_model\data\results\non_partial'
    data_rmse = np.load(r'{}\CHI_result_{}.npy'.format(path, 'rmse'))
    data_rank = np.load(r'{}\CHI_result_{}.npy'.format(path, 'rank'))
    '''
    if select_type == 'results':
        for i in range(0, results.shape[0]):
            results[i, 0] = np.min(data_rmse[i*step:(i+1)*step, 4])
            results[i, 1] = np.min(data_rmse[i*step:(i+1)*step, 5])
            results[i, 2] = np.min(data_rmse[i*step:(i+1)*step, 6])
            results[i, 3] = np.min(data_rmse[i*step:(i+1)*step, 7])
            results[i, 4] = np.max(data_rank[i*step:(i+1)*step, 4])
            results[i, 5] = np.max(data_rank[i*step:(i+1)*step, 5])
            results[i, 6] = np.max(data_rank[i*step:(i+1)*step, 6])
            results[i, 7] = np.max(data_rank[i*step:(i+1)*step, 7])
            results[i, 8] = np.max(data_rank[i*step:(i+1)*step, 8])
            results[i, 9] = np.max(data_rank[i*step:(i+1)*step, 9])
            results[i, 10] = np.max(data_rank[i*step:(i+1)*step, 10])
            results[i, 11] = np.max(data_rank[i*step:(i+1)*step, 11])   
    #elif select_type == 'likelihood':
    #    for i in range(0, 20):
            
    start_id = time_window[0]
    end_id = time_window[1]
    rmse_1 = np.mean(results[start_id:end_id, 0])
    rmse_7 = np.mean(results[start_id:end_id, 1])
    rmse_14 = np.mean(results[start_id:end_id, 2])
    rmse_21 = np.mean(results[start_id:end_id, 3])
    rank_1_5 = np.mean(results[start_id:end_id, 4])
    rank_1_10 = np.mean(results[start_id:end_id, 5])
    rank_7_5 = np.mean(results[start_id:end_id, 6])
    rank_7_10 = np.mean(results[start_id:end_id, 7])
    rank_14_5 = np.mean(results[start_id:end_id, 8])
    rank_14_10 = np.mean(results[start_id:end_id, 9])
    rank_21_5 = np.mean(results[start_id:end_id, 10])
    rank_21_10 = np.mean(results[start_id:end_id, 11])
    
    return results, rmse_1, rmse_7, rmse_14, rmse_21, rank_1_5, rank_1_10, rank_7_5, rank_7_10, rank_14_5, rank_14_10, rank_21_5, rank_21_10
            

def read_comparison_reuslts(model_name, time_window):
    
    path = r'D:/yifei/Documents/Codes_on_GitHub/External_data/CHI_Region'
    #path = r'D:/yifei/Documents/Codes_on_GitHub/External_data/BJ_Grids'
    results = []
    data_file = open('{}/results_{}.txt'.format(path, model_name))
    data_lines = data_file.readlines()
    for data_line in data_lines:
        data = data_line.split('\t')
        tmp = [np.double(data[9]), np.double(data[10]), np.double(data[11]), np.double(data[12]),
               np.double(data[13]), np.double(data[14]), np.double(data[15]), np.double(data[16]),
               np.double(data[17]), np.double(data[18]), np.double(data[19]), np.double(data[20])]
        results.extend([tmp])
    results = np.array(results)
    
    start_id = time_window[0]
    end_id = time_window[1]    
    rmse_1 = np.mean(results[start_id:end_id, 0])
    rmse_7 = np.mean(results[start_id:end_id, 1])
    rmse_14 = np.mean(results[start_id:end_id, 2])
    rmse_21 = np.mean(results[start_id:end_id, 3])
    rank_1_5 = np.mean(results[start_id:end_id, 4])
    rank_1_10 = np.mean(results[start_id:end_id, 5])
    rank_7_5 = np.mean(results[start_id:end_id, 6])
    rank_7_10 = np.mean(results[start_id:end_id, 7])
    rank_14_5 = np.mean(results[start_id:end_id, 8])
    rank_14_10 = np.mean(results[start_id:end_id, 9])
    rank_21_5 = np.mean(results[start_id:end_id, 10])
    rank_21_10 = np.mean(results[start_id:end_id, 11])
    
    return results, rmse_1, rmse_7, rmse_14, rmse_21, rank_1_5, rank_1_10, rank_7_5, rank_7_10, rank_14_5, rank_14_10, rank_21_5, rank_21_10
    
def relation_likelihood_rmse():
    
    m_scaler = sp.MinMaxScaler()
    plt.figure()
    relation = np.zeros([20, 2])
    path = r'D:\yifei\Documents\Codes_on_GitHub\Clustered_CCRF_model\data\results\partial'
    data_rmse = np.load(r'{}\result_{}.npy'.format(path, 'rmse'))
    for i in range(0, 20):
        c, p = ss.pearsonr(data_rmse[i*25:(i+1)*25, 3], data_rmse[i*25:(i+1)*25, 7])
        relation[i, 0] = c
        relation[i, 1] = p
        if p < 0.01:
            plt.plot(m_scaler.fit_transform(data_rmse[i*25:(i+1)*25, 3].reshape([25, 1])), 
                     m_scaler.fit_transform(data_rmse[i*25:(i+1)*25, 7].reshape([25, 1])), 'b*')
    return relation
    
    
    
    
    
    