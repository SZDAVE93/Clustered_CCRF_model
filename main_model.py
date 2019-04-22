# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:36:44 2018

@author: yifei
"""
import os
import sys
import models
import CCRF
import numpy as np
import tree_structure_clustering
import base_line_methods
import build_relation_y2y
import sklearn.mixture as skmix
from sklearn.cluster import KMeans
sys.path.insert(0, 'C:/Users/yifei/Desktop/code/Deep-CRF/')
from DEC_model import DEC
import threading
import time
import feature_test
import DeepNet

class MyThread(threading.Thread):
    
    def __init__(self, thread_id, len_clique, CCRF_X, CCRF_Y, CCRF_S, max_iter,
                 learn_rate, Multi_beta, start_date, train_days, eval_days, lag_days):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.len_clique = len_clique
        self.CCRF_X = CCRF_X
        self.CCRF_Y = CCRF_Y
        self.CCRF_S = CCRF_S
        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.Multi_beta = Multi_beta
        self.start_date = start_date
        self.train_days = train_days
        self.eval_days = eval_days
        self.lag_days = lag_days
        
    def run(self):
        print("clique\t%d\t# of instance:\t%d" %(self.thread_id, self.len_clique))
        self.alpha, self.beta, self.log_likelihood, self.eval_set = models.tuning_model(
                self.CCRF_X, self.CCRF_Y, self.CCRF_S, self.max_iter, self.learn_rate,
                self.Multi_beta, self.start_date, self.train_days, self.eval_days,
                self.lag_days)

def clustered_CCRF_train(m_cliques, m_indexs, start_date, train_days, eval_days, CCRF_X, CCRF_Y, CCRF_S, lag_days):
    
    #start_date = 400
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
    time_start = time.time()
    for cluster_id in range(0, n_cluster):
        clique = m_cliques[cluster_id]
        #t_index = m_indexs[cluster_id][clique, :][:, clique]
        if len(clique) <= 5:
            learn_rate = 1e-2
        elif len(clique) > 5:
            learn_rate = 1e-4
        # training model on each clique
        print("clique\t%d\t# of instance:\t%d" %(cluster_id, len(clique)))
        alpha, beta, log_likelihood, eval_sets = models.tuning_model(CCRF_X[:, clique, :],
                CCRF_Y[clique, :], CCRF_S[:, clique, :][:, :, clique], max_iter, learn_rate,
                Multi_beta, start_date, train_days, eval_days, lag_days)
        paras_alpha[:, cluster_id] = alpha[:, 0]
        paras_beta.extend([beta])
        likelihood = likelihood + log_likelihood
        Eval_sets.extend([eval_sets])
    time_end = time.time()
    print(time_end - time_start)
        
    return Eval_sets, paras_alpha, paras_beta

def clustered_CCRF_eval(Eval_sets, paras_alpha, paras_beta, m_cliques, m_indexs, eval_days):
    
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
        #t_index = m_indexs[cluster_id][clique, :][:, clique]
        e_ccrf_x = Eval_sets[cluster_id][0]
        e_ccrf_y = Eval_sets[cluster_id][1]
        e_ccrf_s = Eval_sets[cluster_id][2]
        alpha = paras_alpha[:, cluster_id].reshape([AR_days, 1])
        beta = paras_beta[cluster_id]
        for i in range(0, eval_days):
            #t_S = e_ccrf_s[i]*t_index*0.2
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
    len_simi = similarity.shape[0]
    indexs = np.zeros([m_number, len_simi, len_simi])
    for i in range(0, m_number):
        t_graph = m_subgraphs[i]
        graph_adj = t_graph.adjacency()
        for adj in graph_adj:
            if len(adj[1]) > 0:
                for ids in adj[1]:
                    indexs[i, adj[0], ids] = 1
    
    return m_cliques, indexs

def cluster_DEC(n_cluster, D_hidden, update_step, epcho, dec_lr, threshold):
    
    raw_data = np.load('C:/Users/yifei/Desktop/code/cofactor-master/fei/co-predict/data/python/X_static_feature.npy')
    raw_data = np.log10(raw_data[:, 15:20] + 1)
    m_DEC = DEC.train(raw_data, n_cluster, D_hidden, update_step, epcho, dec_lr, threshold)
    m_cliques = DEC.predict(m_DEC, raw_data)
    return m_cliques

def cluster_Kmeans(n_cluster, start_date):
    
    raw_data = np.load('C:/Users/yifei/Desktop/code/cofactor-master/fei/co-predict/data/python/CCRF_Y.npy')
    n, t = raw_data.shape
    raw_data = raw_data[:, start_date-300:start_date]
    m_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    len_feature = len(m_bins)
    regions_feature = np.zeros([n, len_feature-1])
    for i in range(0, n):
        regions_feature[i, :] = np.histogram(raw_data[i, :], bins=m_bins)[0] + 0.1
        regions_feature[i, :] = regions_feature[i, :] / np.sum(regions_feature[i, :])
    m_kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(regions_feature)
    regions_label = m_kmeans.labels_
    m_cliques = []
    for i in range(n_cluster):
        temp = np.where(regions_label == i)[0]
        m_cliques.extend([temp])
    return m_cliques

def cluster_Mixture(n_cluster, start_date):
    
    raw_data = np.load('C:/Users/yifei/Desktop/code/cofactor-master/fei/co-predict/data/python/CCRF_Y.npy')
    n, t = raw_data.shape
    raw_data = raw_data[:, start_date-300:start_date]
    m_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    len_feature = len(m_bins)
    regions_feature = np.zeros([n, len_feature-1])
    for i in range(0, n):
        regions_feature[i, :] = np.histogram(raw_data[i, :], bins=m_bins)[0] + 0.1
        regions_feature[i, :] = regions_feature[i, :] / np.sum(regions_feature[i, :])
    m_gaussian = skmix.GaussianMixture(n_cluster)
    m_gaussian.fit(regions_feature)
    regions_label = m_gaussian.predict(regions_feature)
    m_cliques = []
    for i in range(n_cluster):
        temp = np.where(regions_label == i)[0]
        m_cliques.extend([temp])
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
    parser.add_argument('--AR_days', type=int, default=7,
                        help='num of AutoRegreesion days (default: 4)')
    parser.add_argument('--lag_days', type=int, default=1,
                        help='num of lag days (default: 1)')
    parser.add_argument('--child_num', type=int, default=4,
                        help='num of child (default: 4)')
    parser.add_argument('--tree_level', type=int, default=7,
                        help='num of tree level (default: 4)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='value of weight alpha (default: 0.1)')
    parser.add_argument('--crime_type', type=str, default='person',
                        help='value of crime type (default: person)')
    args = parser.parse_args()
    start_date = args.start_date
    train_days = args.train_days
    eval_days = args.eval_days
    AR_days = args.AR_days
    lag_days = args.lag_days
    child_num = args.child_num
    tree_level = args.tree_level
    crime_type = args.crime_type
    alpha = args.alpha
    # CCRF parameters
    #start_date = 600
    #train_days = 30
    #eval_days = 1
    mobility = 0
    #AR_days = 4
    #simi_len = 28*2
    
    # tree-clustering parameters
    child_num = 3
    tree_level = 4
    is_asc = True # indicating whether select the node in asc order or not
    m_type = 0 # 0 for degree based; 1 for edge based
    simi_len = 200
    
    # DEC parameters 
    n_cluster = 5
    D_hidden = [200, 50, 30, 2]
    update_step = 10
    epcho = 10000
    dec_lr = 1e-3
    threshold = 0.02
    
    '''
    X = np.load('C:/Users/yifei/Desktop/code/cofactor-master/fei/co-predict/data/python/X.npy')
    person_crime = [1, 2] # assault and battery
    property_crime = [28, 31] # burglary robber and theft
    if crime_type == 'person':
        X = X[:, :, person_crime]
    else:
        X = X[:, :, property_crime]
    X = np.sum(X, axis=2)
    for AR_days in range(7, 8):
        CCRF_X, CCRF_Y = CCRF.build_data_x2y(X, 0, 0, AR_days)
        CCRF_S = CCRF.build_data_S(CCRF_Y)
        np.save('C:/Users/yifei/Desktop/code/cofactor-master/fei/co-predict/data/paper/tran/Raw_data/CCRF_X_' + str(AR_days) + '_' + crime_type + '.npy', CCRF_X)
        np.save('C:/Users/yifei/Desktop/code/cofactor-master/fei/co-predict/data/paper/tran/Raw_data/CCRF_Y_' + str(AR_days) + '_' + crime_type + '.npy', CCRF_Y)
        np.save('C:/Users/yifei/Desktop/code/cofactor-master/fei/co-predict/data/paper/tran/Raw_data/CCRF_S_' + str(AR_days) + '_' + crime_type + '.npy', CCRF_S)
    '''
    
    myfile = open('C:/Users/yifei/Desktop/results_person_NY_15.txt', 'w')
    #CCRF_X = np.load('C:/Users/yifei/Desktop/code/cofactor-master/fei/co-predict/data/paper/tran/Raw_data/CCRF_X_' + str(AR_days) + '_' + crime_type + '.npy')
    #CCRF_Y = np.load('C:/Users/yifei/Desktop/code/cofactor-master/fei/co-predict/data/paper/tran/Raw_data/CCRF_Y_' + str(AR_days) + '_' + crime_type + '.npy')
    #CCRF_S = np.load('C:/Users/yifei/Desktop/code/cofactor-master/fei/co-predict/data/paper/tran/Raw_data/CCRF_S_' + str(AR_days) + '_' + crime_type + '.npy')
    CCRF_X = np.load('C:/Users/yifei/Desktop/code/Deep-CRF/data/A_crime_property_x_NY.npy')
    CCRF_Y = np.load('C:/Users/yifei/Desktop/code/Deep-CRF/data/A_crime_property_y_NY.npy')
    CCRF_S = np.load('C:/Users/yifei/Desktop/code/Deep-CRF/data/A_crime_property_s_NY.npy')
    simi_others = np.load('C:/Users/yifei/Desktop/code/cofactor-master/fei/co-predict/data/python/simi_poi.npy')
    similarity = build_relation_y2y.build_data_S(CCRF_Y[:, start_date-simi_len:start_date])[-1]
    '''
    alpha = 1
    for i in range(0, CCRF_S.shape[0]):
        CCRF_S[i] = np.mean(alpha * CCRF_S[i] + (1-alpha) * simi_others)
    print("crime_type:%s" % crime_type)
    '''
    results = np.zeros([10, 4])
    simi_type = ['simi_mobi.npy', 'simi_dis.npy', 'simi_poi.npy']
    g_kernels = np.zeros([77, 77, len(simi_type)])
    for i in range(0, len(simi_type)):
        g_kernels[:, :, i] = np.load('C:/Users/yifei/Desktop/code/cofactor-master/fei/co-predict/data/python/' + simi_type[i])
    epcho = 10000
    lr_crfRnn = 1e-5
    rnn_layer = 6
    n = g_kernels.shape[0]
    
    models_all = []
    k=0
    m_cliques, m_indexs = cluster_tree(child_num, tree_level, similarity.copy(), m_type, is_asc)
    crf_parameters = np.zeros([10, len(m_cliques), len(simi_type)])
    for start_date in range(500, 600, 10):
        print("child_num:\t%d\ttree_level:\t%d" % (child_num, tree_level))
        #m_cliques, m_indexs = cluster_tree(child_num, tree_level, similarity.copy(), m_type, is_asc)
        Eval_sets, paras_alpha, paras_beta = clustered_CCRF_train(m_cliques, m_indexs, start_date, 
                                                              train_days, eval_days, CCRF_X, CCRF_Y, CCRF_S, lag_days)
        rmse, P, R = clustered_CCRF_eval(Eval_sets, paras_alpha, paras_beta, m_cliques, m_indexs, eval_days)
        # train multi-CRFasRNN
        '''
        Cluster_CRFasRNN = DeepNet.train_multi(CCRF_Y, CCRF_X, m_cliques, similarity, g_kernels, 
                                               epcho, lr_crfRnn, rnn_layer)
        models_all.extend([Cluster_CRFasRNN])
        P_crfRnn = DeepNet.predict_multi(Eval_sets, m_cliques, Cluster_CRFasRNN, g_kernels)
        rmse_crfRnn = np.sqrt(np.sum((P_crfRnn - R)**2)/(n * eval_days))
        '''
        rmse_crfRnn = 0
        print("start_date:\t%d\tCRF_RMSE:\t%.5f\tM_CRFRNN_RMSE:\t%.5f" % (start_date, rmse, rmse_crfRnn))
        results[k, 0] = start_date
        results[k, 1] = rmse
        results[k, 2] = rmse_crfRnn
        k= k+1
    #print("init a & b:\t%.3f\t%.3f" % (Cluster_CRFasRNN[0].cons_a, Cluster_CRFasRNN[0].cons_b))
    #print(simi_type)
    '''
    for n in range(0, k):
        temp_parameters = np.zeros([len(m_cliques), len(simi_type)])
        temp_model = models_all[n]
        for m in range(0, len(m_cliques)):
            temp = list(temp_model[m].linear_layer.parameters())[0].detach().numpy()
            temp_parameters[m, :] = temp
        crf_parameters[n, :, :] = temp_parameters
    '''
    '''
    for i in range(0, k):
        print(results[k, 1])
    print("Alpha:%.2f\tAVE:\t%.5f" % (alpha, np.mean(results[:, 1])))
    #np.save('C:/Users/yifei/Desktop/code/cofactor-master/fei/co-predict/data/paper/tran/s_cluster_' + str(child_num) + '_' + str(start_date) + '.npy', results)
    '''
    
    '''
    k = 0
    r_data = np.zeros([12, 1])
    for start_date in range(505, 510, 1):
        #similarity = build_relation_y2y.build_data_S(CCRF_Y[:, start_date-simi_len:start_date])[-1]
        #m_cliques, m_indexs = cluster_tree(child_num, tree_level, similarity, m_type, is_asc)
        #print("n_clusters:%d" % len(m_cliques))
        m_indexs = 0
        #m_cliques = cluster_DEC(n_cluster, D_hidden, update_step, epcho, dec_lr, threshold)
        #m_cliques = cluster_Mixture(6, start_date)
        #m_cliques = cluster_Kmeans(8, start_date)
        CCRF_X, CCRF_Y = CCRF.build_data_x2y(X, CCRF_M, 0, 4)
        #CCRF_S = build_relation_y2y.build_data_S(CCRF_Y)
        #CCRF_S = CCRF.build_data_S(np.log(CCRF_Y + 0.01))      
            
        rmse = main_run(start_date, train_days, eval_days, mobility, 4, simi_len,
                        child_num, tree_level, n_cluster, D_hidden, update_step, epcho,
                        dec_lr, threshold, CCRF_X, CCRF_Y, CCRF_S, m_type, is_asc, m_cliques, m_indexs,
                        4)
        #print("is_asc:\t%s\tm_type:\t%d" % (is_asc, m_type))
        print("RMSE:\t%.3f" % rmse)
        r_data[k, 0]= rmse
        k = k + 1
            #dynamic = 1
            #Single = False
            #l_RSME, pred = base_line_methods.eval_model('linear', CCRF_X, CCRF_Y, start_date, 
            #                                            train_days, eval_days, dynamic, Single)
            #print("%s\tRSME:\t%.5f" %('linear', np.sqrt(l_RSME)))
    #np.save('C:/Users/yifei/Desktop/code/cofactor-master/fei/co-predict/data/paper/tran/3_71_cluster.npy', r_data)
    '''
    '''
    str_index = ['linear', 'RFR', 'GPR']
    dynamic = 1
    Single = False
    k = 0
    for start_date in range(500, 600, 10):
        l_RSME, pred = base_line_methods.eval_model(str_index[0], CCRF_X, 
                                                    CCRF_Y,
                                                    start_date, train_days, 
                                                    eval_days, dynamic, Single)
        print("Linear RMSE:\t%.5f" % np.sqrt(l_RSME))
        results[k, 3] = np.sqrt(l_RSME)
        k = k + 1
    '''
    '''
    if len(simi_type) == 3:
        myfile.write("%s\t%s\t%s\n" % (simi_type[0], simi_type[1], simi_type[2]))
    else:
        myfile.write("%s\t%s\n" % (simi_type[0], simi_type[1]))
    for i in range(0, k):
        myfile.write("%d\n" % results[i, 0])
        for j in range(0, len(m_cliques)):
            if len(simi_type) == 3:
                myfile.write("%f\t%f\t%f\n" %(crf_parameters[i, j, 0], crf_parameters[i, j, 1],
                                              crf_parameters[i, j, 2]))
            else:
                myfile.write("%f\t%f\n" %(crf_parameters[i, j, 0], crf_parameters[i, j, 1]))
    '''
    for i in range(0, k):
        myfile.write("%.5f\n" % results[i, 1])
    myfile.close()
    
        #print("%d\tCRF:\t%.5f\tM_CRFRNN:\t%.5f\tLinear:\t%.5f" % (results[i, 0], 
        #                                                        results[i, 1], 
        #                                                        results[i, 2],
        #                                                        results[i, 3]))
        #myfile.write("%d\tCRF:\t%.5f\tM_CRFRNN:\t%.5f\tLinear:\t%.5f\n" % (results[i, 0], 
        #                                                        results[i, 1], 
        #                                                        results[i, 2],
        #                                                        results[i, 3]))
    #myfile.close()
    #print("CRF AVE:\t%.5f" % np.mean(results[:, 1]))
    #print("M_CRFRNN AVE:\t%.5f" % np.mean(results[:, 2]))
    #print("Linear AVE:\t%.5f" % np.mean(results[:, 3]))
    #myfile.write("child_num:\t%d\ttree_level:\t%d\n" % (child_num, tree_level))
    #myfile.write("CRF:\t%.5f\tCRFRNN:\t%.5f\tLinear:\t%.5f\n" %(np.mean(results[:, 1]),
    #                                                            np.mean(results[:, 2]),
    #                                                            np.mean(results[:, 3])))
    #myfile.close()
    
    '''
    for clique in m_cliques:
        l_RSME, pred = base_line_methods.eval_model(str_index[0], CCRF_X[:, clique, :], 
                                                            CCRF_Y[clique, :], start_date, 
                                                            train_days, eval_days, dynamic, Single)
        t = t + l_RSME * len(clique)
        t_rmse[i, 0] = np.sqrt(np.sum((P[clique] - R[clique])**2)/len(clique))
        t_rmse[i, 1] = np.sqrt(l_RSME)
        i = i + 1
    for i in range(0, len(m_cliques)):
        print("%.5f\t%.5f" %(t_rmse[i, 1], t_rmse[i, 0]))
    print("%.5f\t%.5f" % (np.sqrt(t/77), rmse))
    '''
    