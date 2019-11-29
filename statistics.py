# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:23:34 2019

@author: yifei
"""

import seaborn
import functions
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.spatial import distance as dist


def difference(data, length, regions):
    
    diff = []
    for i in range(length, len(data)):
        diff.extend([np.mean(np.abs(data[i] - data[i-length]))])
    diff = np.array(diff)
    mean = np.mean(diff)
    std = np.std(diff)
    return mean, std

def all_difference(data, regions, times):
    
    values = np.zeros([times, 2])
    for i in range(1, times+1):
        mean, std = difference(data, i, regions)
        values[i-1, 0] = mean
        values[i-1, 1] = std
    
    return values

def distance():
    
    results = np.zeros([100, 100])
    for i in range(0, 100-1):
        for j in range(i+1, 100):
            i_x = int(i/10)
            i_y = i%10
            j_x = int(j/10)
            j_y = j%10
            results[i, j] = dist.euclidean([i_x, i_y], [j_x, j_y])
            results[j, i] = results[i, j]
    return results

def checkin_simi(data):
    
    results = np.zeros([100, 100])
    for i in range(0, 100):
        for j in range(i, 100):
            max_value = np.max([np.max(data[:, i]),np.max(data[:, j])]).astype(int)
            results[i, j] = functions.KL_distance(data[:, i], data[:, j], max_value)
            results[j, i] = results[i, j]
    return results

def simi_significant(simi, dist):
    
    results = np.zeros([100, 2])
    for i in range(0, 100):
        c, p = ss.pearsonr(simi[i, :], dist[i, :])
        results[i, 0] = c
        results[i, 1] = p
    return results

def draw_heatmap():
    
    data = np.load(r'D:\yifei\Documents\Codes_on_GitHub\Clustered_CCRF_model\data\CHI\spatial_averaged.npy')
    font = {'family' : 'Calibri',
            'weight' : 'bold',
            'size' : 46}
    annot_font = {'size' : 20,
                  'weight' : 'normal',
                  'color': 'black'}
    
    f, ax = plt.subplots(figsize=(20, 16))
    h = seaborn.heatmap(data, annot=True, annot_kws=annot_font, linewidths=0.5, cbar=False, cmap='Oranges')
    cb=h.figure.colorbar(h.collections[0]) #显示colorbar
    cb.ax.tick_params(labelsize=36) #设置colorbar刻度字体大小。
    cb.ax.set_ylabel('Averaged check-ins', fontdict=font)
    ax.tick_params(axis='y',labelsize=36)
    ax.tick_params(axis='x',labelsize=36)
    ax.set_ylabel('Grid ID', fontdict=font)
    ax.set_xlabel('Grid ID', fontdict=font)
    
def boxplot_spatial_difference():
    
    value_sig = np.load(r'D:\yifei\Documents\Codes_on_GitHub\Clustered_CCRF_model\data\CHI\spatial_significant.npy')
    value = np.load(r'D:\yifei\Documents\Codes_on_GitHub\Clustered_CCRF_model\data\CHI\spatial_insignificant.npy')
    
    f, ax = plt.subplots(2,1,figsize=(20,18), sharex=True)
    
    font = {'family' : 'Calibri',
            'weight' : 'bold',
            'size' : 46}
    line_width = 6
    list_sig = []
    list_nosig = []
    for i in range(0, 5):
        t_sig = np.where((value_sig[:, 0] >= 2*i) & (value_sig[:, 0] < 2*(i+1)))
        t_nosig = np.where((value[:, 0] >= 2*i) & (value[:, 0] < 2*(i+1)))
        t_sig = t_sig[0]
        t_nosig = t_nosig[0]
        list_sig.extend([value_sig[t_sig, 1]])
        list_nosig.extend([value[t_nosig, 1]])
    
    
    ax[0].boxplot(list_sig, widths = 0.5, capprops={'linewidth' : line_width}, whiskerprops={'linewidth' : line_width},
                  medianprops={'linewidth' : line_width, 'color':'red'}, meanprops={'linewidth' : line_width}, 
                  boxprops={'linewidth' : line_width}, 
                  flierprops={'markerfacecolor':'red','markeredgecolor':"black",'markersize':10})
    ax[1].boxplot(list_nosig, widths = 0.5, capprops={'linewidth' : line_width}, whiskerprops={'linewidth' : line_width},
                  medianprops={'linewidth' : line_width, 'color':'red'}, meanprops={'linewidth' : line_width}, 
                  boxprops={'linewidth' : line_width}, 
                  flierprops={'markerfacecolor':'red','markeredgecolor':"black",'markersize':10})
    
    f.text(0.02, 0.61, 'Check-in similarity', fontdict=font, rotation=90)
    
    f.text(0.14, 0.55, 'pcc: -0.55, p_value: 1e-9', fontdict=font, 
           bbox={'facecolor':'white', 'edgecolor':'black', 'linewidth':2})
    ax[0].set_title('Region area A', fontdict=font)
    ax[0].tick_params(axis='y',labelsize=36)
    ax[0].tick_params(axis='x',labelsize=36)
    #ax[0].set_ylabel('Check-in similarity', fontdict=font)
    ax[0].grid(axis='y', ls=':', lw=1, color='gray')
    ax[0].grid(axis='x', ls=':', lw=1, color='gray')
    ax[0].set_ylim((0.9, 1))
    
    ax[0].spines['top'].set_linewidth(2)
    ax[0].spines['right'].set_linewidth(2)
    ax[0].spines['bottom'].set_linewidth(2)
    ax[0].spines['left'].set_linewidth(2)    
    
    f.text(0.14, 0.13, 'pcc: -0.12, p_value: 0.22', fontdict=font,
           bbox={'facecolor':'white', 'edgecolor':'black', 'linewidth':2})
    ax[1].set_title('Region area B', fontdict=font)
    ax[1].set_xticks([1,2,3,4,5])
    ax[1].set_xticklabels(['0~2','2~4','4~6', '6~8', '8~10'])
    ax[1].tick_params(axis='y',labelsize=36)
    ax[1].tick_params(axis='x',labelsize=36)
    #ax[1].set_ylabel('Check-in similarity', fontdict=font)
    ax[1].set_xlabel('Region distance (km)', fontdict=font)
    ax[1].grid(axis='y', ls=':', lw=1, color='gray')
    ax[1].grid(axis='x', ls=':', lw=1, color='gray')
    ax[1].set_ylim((0.9, 1))
    
    ax[1].spines['top'].set_linewidth(2)
    ax[1].spines['right'].set_linewidth(2)
    ax[1].spines['bottom'].set_linewidth(2)
    ax[1].spines['left'].set_linewidth(2)
    
    