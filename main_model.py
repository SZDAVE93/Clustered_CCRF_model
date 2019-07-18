# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:36:44 2018

here again
@author: szdave
"""
import os
import models
import CCRF
import numpy as np
import build_relation_y2y
import tree_structure_clustering

def tuning_model(CCRF_X, CCRF_Y, CCRF_S, max_iter, 
                 learn_rate, Multi_beta, train_start, 
                 train_days, eval_days, lag_days):
    '''
    aim to obtain the best init_alpha and init_beta value to achieve largest likelihood
    return the tuned alpha and beta for prediction
    '''
    init_alpha = 4
    init_beta = 4
    end_alpha = 8
    end_beta = 8
    best_alpha = None
    best_beta = None
    # orgnize train and eval dataset
    Train_sets, Eval_sets = CCRF.build_data_trai_eval(CCRF_X, CCRF_Y, CCRF_S, 
                                                 train_start, train_days, eval_days, lag_days)
    t_ccrf_x = Train_sets[0]
    t_ccrf_y = Train_sets[1]
    t_ccrf_s = Train_sets[2]
    Log_likelihood = -np.inf
    for alpha in range(init_alpha, end_alpha):
        for beta in range(init_beta, end_beta):
            t_alpha, t_beta, log_likelihood = CCRF.fit_model(t_ccrf_x, t_ccrf_y, t_ccrf_s, 
                                                         max_iter, learn_rate, Multi_beta, 
                                                         -alpha, -beta)
            if log_likelihood > Log_likelihood:
                Log_likelihood = log_likelihood
                best_alpha = t_alpha
                best_beta = t_beta
    return best_alpha, best_beta, Log_likelihood, Eval_sets

def clustered_CCRF_train(m_cliques, start_date, train_days, eval_days, CCRF_X, CCRF_Y, CCRF_S, lag_days):

    print(start_date)
    Multi_beta = False
    max_iter = 10
    n_cluster = len(m_cliques)
    likelihood = 0
    print("Separated Tuning......")
    len_alpha = CCRF_X.shape[2]
    paras_alpha = np.zeros([len_alpha, n_cluster])
    paras_beta = []
    Eval_sets = []
    m_threads = []
    for cluster_id in range(0, n_cluster):
        clique = m_cliques[cluster_id]
        # choosing the suitable learning rate
        if len(clique) <= 5:
            learn_rate = 1e-2
        elif len(clique) > 5:
            learn_rate = 1e-4
        # training model on each clique
        print("clique\t%d\t# of instance:\t%d" %(cluster_id, len(clique)))
        alpha, beta, log_likelihood, eval_sets = tuning_model(CCRF_X[:, clique, :],
                CCRF_Y[clique, :], CCRF_S[:, clique, :][:, :, clique], max_iter, learn_rate,
                Multi_beta, start_date, train_days, eval_days, lag_days)
        paras_alpha[:, cluster_id] = alpha[:, 0]
        paras_beta.extend([beta])
        likelihood = likelihood + log_likelihood
        Eval_sets.extend([eval_sets])
        
    return Eval_sets, paras_alpha, paras_beta

def clustered_CCRF_eval(Eval_sets, paras_alpha, paras_beta, m_cliques, eval_days):
    
    # evaluate the model on different region cluster
    print("Evaluating model......")
    Single = False
    Multi_beta = False
    n_clusters = len(m_cliques)    
    AR_days = paras_alpha.shape[0]
    n = 0
    for i in range(0, n_clusters):
        n = n + len(m_cliques[i])
    print("# of areas:\t%d" % n)
    Predicts = np.zeros([n, eval_days])
    Reals = np.zeros([n, eval_days])
    
    print("# of clusters:\t%d" % n_clusters)
    for cluster_id in range(0, n_clusters):
        clique = m_cliques[cluster_id]
        e_ccrf_x = Eval_sets[cluster_id][0]
        e_ccrf_y = Eval_sets[cluster_id][1]
        e_ccrf_s = Eval_sets[cluster_id][2]
        alpha = paras_alpha[:, cluster_id].reshape([AR_days, 1])
        beta = paras_beta[cluster_id]
        for i in range(0, eval_days):
            predict_Y = CCRF.predict(e_ccrf_x[i], e_ccrf_s[i], alpha, beta, Multi_beta)
            Predicts[clique, i] = predict_Y[:, 0]
            Reals[clique, i] = e_ccrf_y[:, i]
    RMSE = np.sqrt(np.sum((Predicts - Reals)**2) / n / eval_days)
    
    return RMSE, Predicts, Reals

def cluster_tree(child_num, tree_level, similarity, m_type, is_asc):
    
    m_cliques, m_number, m_subgraphs = tree_structure_clustering.discover_trees(similarity, 
                                                                                child_num, 
                                                                                tree_level, m_type, 
                                                                                is_asc)    
    return m_cliques


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=int, default=400,
                        help='date of start day (default: 400)')
    parser.add_argument('--train_days', type=int, default=30,
                        help='num of train days (default: 30)')
    parser.add_argument('--eval_days', type=int, default=1,
                        help='num of eval days (default: 1)')
    # AR_days is fixed to 4, any changing with this could failed the code
    # if what, you should first rearrange the input data based on *_y.npy
    # you can figure out of the data rearranging method from the code, or
    # comment to yifeinwpu@gmail.com to find the way to rearrange the data for different AR_days
    parser.add_argument('--AR_days', type=int, default=4,
                        help='num of AutoRegreesion days (default: 4)')
    parser.add_argument('--lag_days', type=int, default=1,
                        help='num of lag days (default: 1)')
    parser.add_argument('--child_num', type=int, default=3,
                        help='num of child (default: 3)')
    parser.add_argument('--tree_level', type=int, default=7,
                        help='num of tree level (default: 4)')
    args = parser.parse_args()
    start_date = args.start_date
    train_days = args.train_days
    eval_days = args.eval_days
    AR_days = args.AR_days
    lag_days = args.lag_days
    child_num = args.child_num
    tree_level = args.tree_level
    mobility = 0
    
    # tree-clustering parameters
    is_asc = True # indicating whether select the node in asc order or not
    m_type = 0 # 0 for degree based; 1 for edge based
    simi_len = 200 # used to specify the similarity between regions and then to cluster
    crime_type = ["person", "property"]
    city_name = ["NY", "CHI"]
    crime_ID = 0
    city_ID = 0
    
    
    path = os.getcwd()
    CCRF_X = np.load(path + '/data/' + city_name[city_ID] + '/' + crime_type[crime_ID] + '_x.npy')
    CCRF_Y = np.load(path + '/data/' + city_name[city_ID] + '/' + crime_type[crime_ID] + '_y.npy')
    CCRF_S = np.load(path + '/data/' + city_name[city_ID] + '/' + crime_type[crime_ID] + '_s.npy')
    # evaluation on different start_date
    for start_date in range(500, 600, 10):
        print("child_num:\t%d\ttree_level:\t%d" % (child_num, tree_level))
        similarity = build_relation_y2y.build_data_S(CCRF_Y[:, start_date-simi_len:start_date])[-1]
        m_cliques = cluster_tree(child_num, tree_level, similarity.copy(), m_type, is_asc)
        Eval_sets, paras_alpha, paras_beta = clustered_CCRF_train(m_cliques, start_date, train_days, 
                                                                  eval_days, CCRF_X, CCRF_Y, CCRF_S, lag_days)
        # P represents the predicted values
        # R represents the real values
        rmse, P, R = clustered_CCRF_eval(Eval_sets, paras_alpha, paras_beta, m_cliques, eval_days)
        print("RMSE:\t%.5f" % rmse)
    
