# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:05:20 2019

@author: yifei
"""

import os
import CCRF
import numpy as np
import functions
import main_model
import tree_structure_clustering as TSC

def ranking_MAP(data, result, rank_k):
    
    n, t = data.shape
    m_MAP = np.zeros([rank_k, 1])
    for i in range(0, t):
        sort_data = sorted(enumerate(data[:, i]), key = lambda x:x[1], reverse=True)
        sort_result = sorted(enumerate(result[:, i]), key = lambda x:x[1], reverse=True)
        sort_data = np.array(sort_data)
        sort_result = np.array(sort_result)
        flags = sort_data[:, 0] == sort_result[:, 0]
        flags = flags.reshape([n, 1])
        for j in range(0, rank_k):
            temp = flags[0:j+1].reshape([1, j+1])
            m_MAP[j] = m_MAP[j] + len(np.where(temp == True)[0]) / (j+1)
    m_MAP = m_MAP / t
    return m_MAP

def optimal_cluster_tune_model(CCRF_X, CCRF_Y, var_weight, start_Date, simi_len, train_days, eval_days):
    
    is_asc = True 
    m_type = 0 
    similarity = functions.build_data_Static_S(CCRF_Y[:, start_Date-simi_len:start_Date])
    CCRF_S = functions.transfer_static_S(similarity, CCRF_Y.shape[1])
    
    results = np.zeros([1, 8])
    likelihood = -np.inf
    k = 0
    for child_num in range(6, 7):
        for tree_level in range(2, 3):
            #if child_num > 2:
            #    tree_level = 2
            '''
            m_cliques, m_number, m_subgraphs = TSC.discover_trees(similarity.copy(), 
                                                                  child_num, 
                                                                  tree_level, m_type, 
                                                                  is_asc, var_weight)
            m_modularity, m_graph = functions.clique_modularity(m_cliques, 
                                                                similarity.copy(), 
                                                                var_weight)
            p_CCRF_S, zero_ratio = functions.fully2partial(CCRF_S.copy(), m_graph)
            '''
            m_cliques = [[x for x in range(0, 100)]]
            p_CCRF_S = CCRF_S.copy()
            Eval_sets, alpha, beta, t_likelihood = main_model.clustered_CCRF_train(m_cliques, start_Date, train_days, 
                                                                      eval_days, CCRF_X, CCRF_Y, p_CCRF_S, 1)
            c_rmse, c_rmse_iter, c_rmse_iter_1, c_rmse_iter_7, c_rmse_iter_14, P, P_y, R = main_model.clustered_CCRF_eval(Eval_sets, alpha, beta, m_cliques, eval_days)
            '''
            # for ranking
            print("child_num:\t{}\ttree_level:\t{}\tnum:\t{}".format(child_num, tree_level, len(m_cliques)))
            rankings_1 = ranking_MAP(R[:, 0:1], P_y[:, 0:1], 10)
            rankings_7 = ranking_MAP(R[:, 0:7], P_y[:, 0:7], 10)
            rankings_14 = ranking_MAP(R[:, 0:14], P_y[:, 0:14], 10)
            rankings_21 = ranking_MAP(R[:, 0:21], P_y[:, 0:21], 10)
            results[k, 0] = len(m_cliques)
            results[k, 1] = child_num
            results[k, 2] = tree_level
            results[k, 3] = t_likelihood[0, 0]
            results[k, 4] = np.mean(rankings_1[0:5])
            results[k, 5] = np.mean(rankings_1[0:10])
            results[k, 6] = np.mean(rankings_7[0:5])
            results[k, 7] = np.mean(rankings_7[0:10])
            results[k, 8] = np.mean(rankings_14[0:5])
            results[k, 9] = np.mean(rankings_14[0:10])
            results[k, 10] = np.mean(rankings_21[0:5])
            results[k, 11] = np.mean(rankings_21[0:10])
            '''
            # for prediction
            print("child_num:\t{}\ttree_level:\t{}\tnum:\t{}".format(child_num, tree_level, len(m_cliques)))
            print("Log_likelihood:\t{:.5f}\tRMSE_ITER:\t{:.5f}".format(t_likelihood[0, 0], c_rmse_iter))
            print("RMSE_ITER_1:\t{:.5f}\tRMSE_ITER_7:\t{:.5f}".format(c_rmse_iter_1, c_rmse_iter_7))
            results[k, 0] = len(m_cliques)
            results[k, 1] = child_num
            results[k, 2] = tree_level
            results[k, 3] = t_likelihood[0, 0]
            results[k, 4] = c_rmse_iter_1
            results[k, 5] = c_rmse_iter_7
            results[k, 6] = c_rmse_iter_14
            results[k, 7] = c_rmse_iter
            
            if t_likelihood[0, 0] > likelihood:
                likelihood = t_likelihood[0, 0]
                likelihood_measure = [c_rmse_iter_1, c_rmse_iter_7, c_rmse_iter_14, c_rmse_iter]
                #likelihood_measure = [results[k, 4], results[k, 5], results[k, 6], results[k, 7],
                #                      results[k, 8], results[k, 9], results[k, 10], results[k, 11]]
                cliques = m_cliques
                m_alpha = alpha
                m_beta = beta
            k = k + 1
            #if child_num > 2:
            #    break
    return results, likelihood, likelihood_measure, cliques, m_alpha, m_beta

if __name__ == "__main__":

    AR_days = 7
    var_weight = 2
    #start_Date = 760
    simi_len = 30
    train_days = 30
    eval_days = 21
    path = os.getcwd()
    # CHI data
    '''
    X = np.load(path + '/data/CHI/raw/crime_record.npy')
    M = np.load(path + '/data/CHI/raw/region_factors.npy')
    '''
    # BJ data
    X = np.load(path + '/data/BJ/raw/checkin_record.npy')
    M = 0
    
    m_tag = 0 # not using M(region factors) as input factors
    #CCRF_X, CCRF_Y = CCRF.build_data_x2y_new(X, m_tag, AR_days)
    CCRF_X, CCRF_Y = CCRF.build_data_x2y(X, M, m_tag, AR_days)
    result = np.zeros([1, 8])
    m_cliques = {}
    alphas = {}
    betas = {}
    likelihood_rmses = {}
    for start_Date in range(31, 60, 10):
        results, likelihood, likelihood_measure, cliques, m_alpha, m_beta = optimal_cluster_tune_model(CCRF_X, 
                                                                                     CCRF_Y, 
                                                                                     var_weight, 
                                                                                     start_Date, 
                                                                                     simi_len, 
                                                                                     train_days, 
                                                                                     eval_days)
        result = np.append(result, results, axis=0)
        m_cliques.update({start_Date:cliques})
        alphas.update({start_Date:m_alpha})
        betas.update({start_Date:m_beta})
        likelihood_rmses.update({start_Date:[likelihood, likelihood_measure]})
    result = result[1:]
    np.save(r'{}\data\results\non_partial\BJ_result_rmse.npy'.format(path), result)
    '''
    np.save(r'{}\data\results\partial\BJ_likelihood_rank_all.npy'.format(path), result)
    np.save(r'{}\data\results\partial\BJ_likelihood_rank_cliques.npy'.format(path), m_cliques)
    np.save(r'{}\data\results\partial\BJ_likelihood_rank_alphas.npy'.format(path), alphas)
    np.save(r'{}\data\results\partial\BJ_likelihood_rank_betas.npy'.format(path), betas)
    np.save(r'{}\data\results\partial\BJ_likelihood_ranks.npy'.format(path), likelihood_measure)
    '''
    