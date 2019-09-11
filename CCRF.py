# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 22:28:09 2018

@author: szdave
"""
import os
import numpy as np
import scipy.spatial.distance as dist


def calculate_gradient_alpha(X, Y, S, log_alpha, k_th, A, b):
    '''
    get k-th gradient for k-th log_alpha
    '''
    t, n, m = X.shape
    gradient = 0
    
    alpha = np.exp(log_alpha)
    inv_A = np.zeros([t, n, n])
    for i in range(0, t):
        try:
            inv_A[i] = np.linalg.inv(A[i])
        except:
            inv_A[i] = np.linalg.pinv(A[i])
    I = np.eye(n).flatten().reshape([n**2, 1])
    for i in range(0, t):
        term_1 = np.sum((Y[:, i].reshape([n, 1]) - X[i, :, k_th])**2)
        term_2 = - 1/2 * np.dot(inv_A[i].flatten().reshape([n**2, 1]).T, I)
        term_3 = np.dot(np.dot(X[i, :, k_th].reshape([n, 1]).T, inv_A[i]), b[:, i].reshape([n, 1]))
        term_4 = - np.dot(np.dot(b[:, i].reshape([n, 1]).T, inv_A[i]), np.dot(inv_A[i], b[:, i].reshape([n, 1])))
        term_5 = np.dot(np.dot(b[:, i].reshape([n, 1]).T, inv_A[i]), X[i, :, k_th].reshape([n, 1]))
        term_6 = - np.sum((X[i, :, k_th].reshape([n, 1]))**2)
        gradient = gradient + term_1 + term_2 + term_3 + term_4 + term_5 + term_6
    gradient = - alpha[k_th] * gradient
    
    return gradient


def do_gradient_alpha(X, Y, S, log_alpha, A, b):
    '''
    graident for each log_alpha
    '''
    t, n, m = X.shape
    gradients = np.zeros([m, 1])
    
    for i in range(0, m):
        gradients[i] = calculate_gradient_alpha(X, Y, S, log_alpha, i, A, b)
    
    return gradients

def calculate_gradient_beta(X, Y, S, log_beta, k_th, A, b):
    '''
    get k-th gradient for k-th log_beta
    '''
    t, n, m = X.shape
    
    beta = np.exp(log_beta)
    inv_A = np.zeros([t, n, n])
    for i in range(0, t):
        try:
            inv_A[i] = np.linalg.inv(A[i])
        except:
            inv_A[i] = np.linalg.pinv(A[i])
    #I = np.eye(n).flatten().reshape([n**2, 1])
    gradient = 0
    for i in range(0, t):
        Y_t = Y[:, i].reshape([n, 1])
        temp = np.ones([n, n])
        temp = (Y_t - Y_t.T * temp)**2
        term_1 = 1/2 * np.sum(S[i] * temp)
        # here, we should make sure that S[i, i] = 0
        D = np.sum(S[i], axis=1).reshape([n, 1]) * np.eye(n)
        temp = np.zeros([n, 1])
        temp[k_th] = 1
        D_S_k_1 = (D - S[i]) * temp
        D_S_K_2 = np.diag(S[i, k_th, :])
        D_S_k = D_S_k_1 + D_S_K_2
        term_2 = - 1/2 * np.dot(inv_A[i].flatten().reshape([n**2, 1]).T, D_S_k.T.flatten().reshape([n**2, 1]))
        term_3 = - np.dot(np.dot(np.dot(b[:, i].reshape([n, 1]).T, inv_A[i]), D_S_k), np.dot(inv_A[i], b[:, i].reshape([n, 1])))
        gradient = gradient + term_1 + term_2 + term_3
    gradient = - beta[k_th] * gradient
    #print(gradient.shape)
    return gradient

def do_gradient_beta(X, Y, S, log_beta, A, b, Multi_b):
    '''
    gradient for beta
    '''
    t, n, m = X.shape
    
    if Multi_b:
        gradient = np.zeros([n, 1])
        for i in range(0, n):
            gradient[i] = calculate_gradient_beta(X, Y, S, log_beta, i, A, b)
    else:    
        beta = np.exp(log_beta)
        inv_A = np.zeros([t, n, n])
        for i in range(0, t):
            try:
                inv_A[i] = np.linalg.inv(A[i])
            except:
                inv_A[i] = np.linalg.pinv(A[i])
        #I = np.eye(n).flatten().reshape([n**2, 1])
        gradient = 0
        for i in range(0, t):
            Y_t = Y[:, i].reshape([n, 1])
            temp = np.ones([n, n])
            temp = (Y_t - Y_t.T * temp)**2
            term_1 = 1/2 * np.sum(S[i] * temp)
            # here, we should make sure that S[i, i] = 0
            D = np.sum(S[i], axis=1).reshape([n, 1]) * np.eye(n)
            D_S = D - S[i]
            term_2 = - 1/2 * np.dot(inv_A[i].flatten().reshape([n**2, 1]).T, D_S.T.flatten().reshape([n**2, 1]))
            term_3 = - np.dot(np.dot(np.dot(b[:, i].reshape([n, 1]).T, inv_A[i]), D_S), np.dot(inv_A[i], b[:, i].reshape([n, 1])))
            gradient = gradient + term_1 + term_2 + term_3
        gradient = - beta * gradient
    
    return gradient
   
def calculate_matrix_A(S, log_alpha, log_beta, Multi_b):
    '''
    calculate A
    '''
    t = S.shape[0]
    n = S.shape[1]
    A = np.zeros([t, n, n])
    alpha = np.exp(log_alpha)
    beta = np.exp(log_beta)
    
    if Multi_b:
        # calculate A for Multi_b
        for i in range(0, t):
            D = np.sum(S[i], axis=1).reshape([n, 1]) * np.eye(n)
            D_S = D - S[i]
            term_1 = np.sum(alpha) * np.eye(n)
            temp = np.diag(np.sum(beta * S[i], axis=0))
            term_2 = beta * D_S + temp
            A[i] = term_1 + term_2
    else:
        # calculate A for single b
        for i in range(0, t):
            D = np.sum(S[i], axis=1).reshape([n, 1]) * np.eye(n)
            D_S = D - S[i]
            term_1 = np.sum(alpha) * np.eye(n)
            term_2 = beta * D_S
            A[i] = term_1 + term_2
    
    return A
      
def calculate_matrix_b(X, log_alpha):
    '''
    calculate b
    '''
    
    t, n, m = X.shape
    b = np.zeros([n, t])
    alpha = np.exp(log_alpha)
    
    for i in range(0, t):
        b[:, i] = np.dot(X[i], alpha)[:, 0]
    
    return b

def calculate_likelihood(X_t, Y_t, S_t, log_alpha, log_beta, A_t, b_t):
    '''
    calculate log_likelihood for each dataset
    '''
    n = Y_t.shape[0]
    alpha = np.exp(log_alpha)
    beta = np.exp(log_beta)
    
    term_1 = - np.sum(np.dot((Y_t - X_t)**2, alpha))
    temp = np.ones([n, n])
    temp = (Y_t - Y_t.T * temp)**2
    term_2 = - 1/2 * np.sum(beta * S_t * temp)
    Z_t_c = np.sum(np.dot(X_t**2, alpha))
    try:
        inv_A_t = np.linalg.inv(A_t)
    except:
        inv_A_t = np.linalg.pinv(A_t)
    Z_t = (np.linalg.det(2*A_t))**(-1/2) * np.exp(np.dot(np.dot(b_t.T, inv_A_t), b_t) - Z_t_c)
    Z_t = (2*np.pi)**(n/2) * Z_t
    term_3 = - np.log(Z_t)
    
    #print("Term_1:\t%.5f" % term_1)
    #print("Term_2:\t%.5f" % term_2)
    #print("Term_3:\t%.5f" % term_3)
    log_likelihood = term_1 + term_2 + term_3
    #print("log_likelihood:\t%.5f" % log_likelihood)
    
    return log_likelihood
    

def fit_model(X, Y, S, max_iter, learn_rate, Multi_beta, init_alpha, init_beta):
    '''
    X: feature sets (t, n, m)
    Y: outputs (n, t)
    S: relation between outputs (t, n, n)
    '''
    t, n, m = X.shape
    init_value_alpha = init_alpha
    init_value_beta = init_beta
    if Multi_beta:
        #print("leanring with multi beta")
        log_beta = init_value_beta * np.ones([n, 1])
    else:
        #print("learning with single beta")
        log_beta = init_value_beta * np.ones([1, 1])
    log_alpha = init_value_alpha * np.ones([m, 1])
    Log_likelihood = -np.inf
    
    for i in range(0, max_iter):
        Matrix_A = calculate_matrix_A(S, log_alpha, log_beta, Multi_beta)
        Matrix_b = calculate_matrix_b(X, log_alpha)
        # calculate likelihood, due to A, b will be update with following processes, so we do calculate here
        log_likelihood = 0
        for i in range(0, t):
            X_t = X[i]
            Y_t = Y[:, i].reshape([n, 1])
            S_t = S[i]
            A = Matrix_A[i]
            b = Matrix_b[:, i].reshape([n, 1])
            log_likelihood = log_likelihood + calculate_likelihood(X_t, Y_t, S_t, log_alpha, log_beta, A, b)
        #print("Log_likelihood:\t%.5f" % log_likelihood)
        #print(log_likelihood.shape)
        if log_likelihood > Log_likelihood:
            #print("Log_likelihood:\t%.5f" % log_likelihood)
            Log_likelihood = log_likelihood
        else:
            #print("except out!")
            break
        gradient_log_alpha = do_gradient_alpha(X, Y, S, log_alpha, Matrix_A, Matrix_b)
        gradient_log_beta = do_gradient_beta(X, Y, S, log_beta, Matrix_A, Matrix_b, Multi_beta)
        #print(gradient_log_alpha[:, 0])
        #print(gradient_log_beta[0])
        #print("alpha:")
        #print(np.exp(log_alpha)[:, 0])
        #print("beta:")
        #print(np.exp(log_beta)[0])
        log_alpha = log_alpha + learn_rate * gradient_log_alpha
        log_beta = log_beta + learn_rate * gradient_log_beta
        #print("new_alpha:")
        #print(np.exp(log_alpha)[:, 0])
        #print("new_beta:")
        #print(np.exp(log_beta)[0])
        alpha = np.exp(log_alpha)
        beta = np.exp(log_beta)
    
    return alpha, beta, Log_likelihood

def predict(X_t, S_t, alpha, beta, Multi_beta):
    '''
    predict Y_t w.r.t S_t and X_t
    '''
    #beta = 0
    if Multi_beta:
        # predict use multi b
        n = X_t.shape[0]
        D = np.sum(S_t, axis=1).reshape([n, 1]) * np.eye(n)
        D_S = D - S_t
        term_1 = np.sum(alpha) * np.eye(n)
        #temp = np.diag(np.sum(beta * S_t, axis=0))
        term_2 = beta * D_S# + temp
        A = term_1 + term_2
        try:
            inv_A = np.linalg.inv(A)
        except:
            inv_A = np.linalg.pinv(A)
        predict_Y = np.dot(inv_A, np.dot(X_t, alpha))
    else:
        # predict use one b
        n = X_t.shape[0]
        D = np.sum(S_t, axis=1).reshape([n, 1]) * np.eye(n)
        D_S = D - S_t
        term_1 = np.sum(alpha) * np.eye(n)
        term_2 = beta * D_S
        A = term_1 + term_2
        try:
            inv_A = np.linalg.inv(A)
        except:
            inv_A = np.linalg.pinv(A)
        predict_Y = np.dot(inv_A, np.dot(X_t, alpha))
    
    return predict_Y

def build_data_x2y_new(X, tag, ar_days):
    
    start_date = 0
    end_date = 365*3
    AR_days = ar_days
    
    X_all = np.sum(X[start_date:end_date], axis=2)
    t, n = X_all.shape
    if tag == 0: # only near historical data
        num_data = end_date - AR_days
        AR_train_all = np.zeros([num_data, n, AR_days+1])
        for i in range(AR_days, end_date):
            temp = X_all[i-AR_days:i+1, :].T
            AR_train_all[i-AR_days] = temp
        CCRF_X = AR_train_all[:, :, 0:AR_days]
        CCRF_Y = AR_train_all[:, :, AR_days].T
    else: # with long term historical data
        num_data = end_date - AR_days - 2 * 7
        AR_train_all = np.zeros([num_data, n, AR_days+1+2])
        for i in range(AR_days + 2*7, end_date):
            temp = X_all[i-AR_days:i+1, :].T
            temp_long = X_all[[i-14, i-21], :].T
            temp_long = np.append(temp_long, temp, axis=1)
            AR_train_all[i-AR_days-2*7] = temp_long
        CCRF_X = AR_train_all[:, :, 0:AR_days+2]
        CCRF_Y = AR_train_all[:, :, AR_days+2].T
    print(CCRF_X.shape)
    return CCRF_X, CCRF_Y

def build_data_x2y(X, M, tag, ar_days):
    '''
    build data that are used for the first feature function, X -> Y
    '''
    start_date = 0
    end_date = 365*3
    AR_days = ar_days # how many historical days are used to predict, here we take 7 days into account
    
    X_all_1 = np.sum(X[start_date:end_date], axis=2)
    
    #X_all_1 = X.copy()
    t, n = X_all_1.shape
    num_data = end_date - AR_days
    
    AR_train_all = np.zeros([num_data, n, AR_days+1])
    for i in range(AR_days, end_date):
        temp = X_all_1[i-AR_days:i+1, :].T
        AR_train_all[i-AR_days] = temp
    
    if tag == 0:
        # only use historical records
        CCRF_X = AR_train_all[:, :, 0:AR_days]
        CCRF_Y = AR_train_all[:, :, AR_days].T
    else:
        # consider region mobility feature from M as well
        CCRF_X = AR_train_all[:, :, 0:AR_days]
        CCRF_Y = AR_train_all[:, :, AR_days].T
        m_start = 27
        m_end = M.shape[2]
        CCRF_XM = np.zeros([num_data, n, AR_days + m_end - m_start])
        for i in range(start_date, start_date+num_data):
            #print(CCRF_X[i].shape)
            #print(M[i, :, m_start:m_end].shape)
            CCRF_XM[i] = np.append(CCRF_X[i], M[i+AR_days-1, :, m_start:m_end], axis=1)
            #CCRF_XM[i] = np.append(CCRF_X[i], np.sum(M[i:i+AR_days-1, :, m_start:m_end], axis=0), axis=1)
        CCRF_X = CCRF_XM
    
        
    return CCRF_X, CCRF_Y

def build_data_S(CCRF_Y):
    '''
    assume the relationship between y is exp(-distance(y_i - y_j))
    when use this data, remember to introduce lag_days
    to make the value in between 0 to 1, we apply sigmoid here
    '''
    n, t = CCRF_Y.shape
    CCRF_S = np.zeros([t, n, n])
    #I = np.eye(n)
    
    for k in range(0, t):
        for i in range(0, n):
            for j in range(i, n):
                t_dist = dist.euclidean(CCRF_Y[i, k], CCRF_Y[j, k])
                temp = 1 / (1 + np.exp(0.2 * (t_dist - 10)))
                CCRF_S[k, i, j] = temp
                CCRF_S[k, j, i] = CCRF_S[k, i, j]
        # make sure that S_{i, j} = 0
        #CCRF_S[k] = CCRF_S[k] - I
    
    return CCRF_S    

def build_data_trai_eval(CCRF_X, CCRF_Y, CCRF_S, train_end, train_days, eval_days, lag_days):
    '''
    build train and eval dataset
    '''
    
    #lag_days = 14 # decide how to apply CCRF_S
    #print("lag_days:%d" % lag_days)
    eval_start = train_end
    
    # train set
    T_CCRF_X = CCRF_X[train_end-train_days:train_end]
    T_CCRF_Y = CCRF_Y[:, train_end-train_days:train_end]
    T_CCRF_S = CCRF_S[train_end-lag_days-train_days:train_end-lag_days]
    
    # eval set
    E_CCRF_X = CCRF_X[eval_start:eval_start+eval_days]
    E_CCRF_Y = CCRF_Y[:, eval_start-1:eval_start+eval_days]
    E_CCRF_S = CCRF_S[eval_start-lag_days:eval_start-lag_days+eval_days]
    
    return (T_CCRF_X, T_CCRF_Y, T_CCRF_S), (E_CCRF_X, E_CCRF_Y, E_CCRF_S)