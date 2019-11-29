# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:36:44 2018

@author: szdave
"""
import os
import CCRF
import numpy as np
import functions
import tree_structure_clustering

def tuning_model(CCRF_X, CCRF_Y, CCRF_S, max_iter, 
                 learn_rate, Multi_beta, train_start, 
                 train_days, eval_days, lag_days):
    '''
    aim to obtain the best init_alpha and init_beta value to achieve largest likelihood
    return the tuned alpha and beta for prediction
    '''
    reg = False
    reg_lambda = 1e6
    init_alpha = 4
    init_beta = 4
    end_alpha = 12
    end_beta = 12
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
                                                         -alpha, -beta, reg, reg_lambda)
            if log_likelihood > Log_likelihood:
                Log_likelihood = log_likelihood
                best_alpha = t_alpha
                best_beta = t_beta
    return best_alpha, best_beta, Log_likelihood, Eval_sets

def clustered_CCRF_train(m_cliques, start_date, train_days, eval_days, CCRF_X, CCRF_Y, CCRF_S, lag_days):

    print(start_date)
    Multi_beta = False
    max_iter = 1
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
            learn_rate = 1e-6
        # training model on each clique
        print("clique\t%d\t# of instance:\t%d" %(cluster_id, len(clique)))
        alpha, beta, log_likelihood, eval_sets = tuning_model(CCRF_X[:, clique, :],
                CCRF_Y[clique, :], CCRF_S[:, clique, :][:, :, clique], max_iter, learn_rate,
                Multi_beta, start_date, train_days, eval_days, lag_days)
        paras_alpha[:, cluster_id] = alpha[:, 0]
        paras_beta.extend([beta])
        likelihood = likelihood + log_likelihood
        Eval_sets.extend([eval_sets])
        
    return Eval_sets, paras_alpha, paras_beta, likelihood

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
    pred_y_all = np.zeros([n, eval_days])
    
    print("# of clusters:\t%d" % n_clusters)
    for cluster_id in range(0, n_clusters):
        clique = m_cliques[cluster_id]
        e_ccrf_x = Eval_sets[cluster_id][0]
        e_ccrf_y = Eval_sets[cluster_id][1]
        e_ccrf_s = Eval_sets[cluster_id][2]
        alpha = paras_alpha[:, cluster_id].reshape([AR_days, 1])
        beta = paras_beta[cluster_id]
        iter_x = e_ccrf_x[0]
        num_regions, seq_len = iter_x.shape
        pred_y = np.zeros([len(clique), 1])
        for i in range(0, eval_days):
            predict_Y = CCRF.predict(e_ccrf_x[i], e_ccrf_s[i], alpha, beta, Multi_beta)
            Predicts[clique, i] = predict_Y[:, 0]
            Reals[clique, i] = e_ccrf_y[:, i]
            pred_y_tmp = CCRF.predict(iter_x, e_ccrf_s[i], alpha, beta, Multi_beta)
            pred_y = np.append(pred_y, pred_y_tmp, axis=1)
            iter_x = np.append(iter_x[:, 1:seq_len], pred_y_tmp, axis=1)
        pred_y = pred_y[:, 1:eval_days+1]
        pred_y_all[clique, :] = pred_y
    RMSE = np.sqrt(np.sum((Predicts - Reals)**2) / n / eval_days)
    RMSE_iter = np.sqrt(np.sum((pred_y_all - Reals)**2) / n / eval_days)
    RMSE_iter_1 = np.sqrt(np.sum((pred_y_all[:, 0] - Reals[:, 0])**2) / n / 1)
    RMSE_iter_7 = np.sqrt(np.sum((pred_y_all[:, 0:7] - Reals[:, 0:7])**2) / n / 7)
    RMSE_iter_14 = np.sqrt(np.sum((pred_y_all[:, 0:14] - Reals[:, 0:14])**2) / n / 14)
    
    return RMSE, RMSE_iter, RMSE_iter_1, RMSE_iter_7, RMSE_iter_14, Predicts, pred_y_all, Reals

def cluster_tree(child_num, tree_level, similarity, m_type, is_asc, var_weight):
    
    m_cliques, m_number, m_subgraphs = tree_structure_clustering.discover_trees(similarity, 
                                                                                child_num, 
                                                                                tree_level, m_type, 
                                                                                is_asc, var_weight)    
    return m_cliques, m_subgraphs


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=int, default=750,
                        help='date of start day (default: 400)')
    parser.add_argument('--train_days', type=int, default=180,
                        help='num of train days (default: 180)')
    parser.add_argument('--eval_days', type=int, default=14,
                        help='num of eval days (default: 1)')
    parser.add_argument('--AR_days', type=int, default=7,
                        help='num of AutoRegreesion days (default: 4)')
    parser.add_argument('--lag_days', type=int, default=1,
                        help='num of lag days (default: 1)')
    parser.add_argument('--child_num', type=int, default=5,
                        help='num of child (default: 3)')
    parser.add_argument('--tree_level', type=int, default=3,
                        help='num of tree level (default: 4)')
    parser.add_argument('--is_fullPartial', type=int, default=1,
                        help='whether to partial the matrix 1/0')
    parser.add_argument('--simi_len', type=int, default=90,
                        help='build similarity matrix between regions')
    args = parser.parse_args()
    start_Date = args.start_date
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
    simi_len = args.simi_len # used to specify the similarity between regions and then to cluster
    crime_type = ["person", "property"]
    city_name = ["NY", "CHI", "BJ"]
    crime_ID = 0
    city_ID = 1
    var_weight = 2
    
    # CCRF_S parameter
    is_fullPartial = args.is_fullPartial
    
    # load existing training dataset
    '''
    path = os.getcwd()
    CCRF_X = np.load(path + '/data/' + city_name[city_ID] + '/sampled/' + crime_type[crime_ID] + '_x.npy')
    CCRF_Y = np.load(path + '/data/' + city_name[city_ID] + '/sampled/' + crime_type[crime_ID] + '_y.npy')
    CCRF_S = np.load(path + '/data/' + city_name[city_ID] + '/sampled/' + crime_type[crime_ID] + '_s.npy')
    n = CCRF_S.shape[1]
    '''
    path = os.getcwd()
    m_tag = 0 # not using M(region factors) as input factors
    if city_ID == 2:
        X = np.load(path + '/data/' + city_name[city_ID] + '/raw/checkin_record.npy')
        M = 0
        CCRF_X, CCRF_Y = CCRF.build_data_x2y(X, M, m_tag, AR_days)
    else:
    # build new training dataset
        X = np.load(path + '/data/' + city_name[city_ID] + '/raw/crime_record.npy')
        M = np.load(path + '/data/' + city_name[city_ID] + '/raw/region_factors.npy')
        CCRF_X, CCRF_Y = CCRF.build_data_x2y_new(X, m_tag, AR_days)
    n = CCRF_Y.shape[0]
    
    if is_fullPartial == 1:
        file = open(path + '/data/results/partial/10results_{}_{}.txt'.format(simi_len, eval_days), 'a+')
    else:
        file = open(path + '/data/results/non_partial/10results_{}_{}.txt'.format(simi_len, eval_days), 'a+')
    # evaluation on different clustering parameters
    for child_num in range(2, 4):
        for tree_level in range(2, 4):
            print("child_num:\t%d\ttree_level:\t%d" % (child_num, tree_level))
            similarity = functions.build_data_Static_S(CCRF_Y[:, start_Date-simi_len:start_Date])
            CCRF_S = functions.transfer_static_S(similarity, CCRF_Y.shape[1])
            
            m_cliques, m_subgraphs = cluster_tree(child_num, tree_level, similarity.copy(), m_type, is_asc, var_weight)
            m_modularity, m_graph = functions.clique_modularity(m_cliques, similarity.copy(), var_weight)
            m_entropy = functions.clique_entropy(m_cliques, CCRF_Y[:, start_Date-train_days:start_Date])
            
            if is_fullPartial == 1:
                print("# of edges:\t{}".format(m_graph.number_of_edges()))
                p_CCRF_S, zero_ratio = functions.fully2partial(CCRF_S.copy(), m_graph)
            else:
                zero_ratio = 0
                p_CCRF_S = CCRF_S.copy()
            
            Eval_sets, paras_alpha, paras_beta, likelihood = clustered_CCRF_train(m_cliques, start_Date, train_days, 
                                                                      eval_days, CCRF_X, CCRF_Y, p_CCRF_S, lag_days)
            # P represents the predicted values
            # R represents the real values
            c_rmse, c_rmse_iter, P, P_y, R = clustered_CCRF_eval(Eval_sets, paras_alpha, paras_beta, m_cliques, eval_days)
            print("Clustered_CCRF\tRMSE:\t%.5f" % c_rmse)
            print("Clustered_CCRF\tRMSE_iter:\t%.5f" % c_rmse_iter)
            file.write("{}\t{}\t{}\t{}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(start_Date, child_num, tree_level, len(m_cliques), 
                                                                                     m_modularity, m_entropy, zero_ratio, 
                                                                                     c_rmse, c_rmse_iter))
    file.close()
    
    
    
        
    
