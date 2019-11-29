# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:18:33 2018

@author: szdave
"""

import numpy as np
import networkx as nx


def searchNeighbor(node_id, graph, child_num, simi_matrix):
    '''
    search neighbors and then add new edges
    '''
    neighbors = list(graph.neighbors(node_id))
    len_neighbors = len(neighbors)
    
    if len_neighbors > child_num:
        results = []
        simi_ids = np.zeros([len_neighbors, 2], dtype=int)
        simi_ids[:, 0] = node_id
        simi_ids[:, 1] = neighbors
        t_simi = simi_matrix[simi_ids[:, 0], simi_ids[:, 1]]
        t_b = sorted(enumerate(t_simi), key = lambda x:x[1])
        for i in range(0, child_num):
            results.extend([neighbors[t_b[len_neighbors-i-1][0]]])
    else:
        results = neighbors
    
    return results

def buildGraph(simi_matrix, node_num, threshold):
    '''
    add nodes to build graph
    '''
    region_flags = np.zeros([1, node_num])
    m_graph = nx.Graph()
    while np.sum(region_flags) < node_num:
        temp = np.where(simi_matrix == np.max(simi_matrix))
        t_edge = [temp[0][0], temp[1][0]]
        m_graph.add_edge(t_edge[0], t_edge[1])
        simi_matrix[t_edge[0], t_edge[1]] = 0
        simi_matrix[t_edge[1], t_edge[0]] = 0
        region_flags[0, t_edge] = 1
    
    return m_graph

def buildGraphS(simi_matrix, threshold):
    '''
    according threshold to remove edges
    '''
    n = simi_matrix.shape[0]
    m_graph = nx.Graph()
    for i in range(0, n-1):
        for j in range(i+1, n):
            m_graph.add_edge(i, j)
    edges = list(m_graph.edges)
    #for edge in edges:
    #    if simi_matrix[]
    
    return m_graph

def checkSingleNodes(graph, simi_matrix):
    '''
    connect single nodes to their most closed neighbors
    '''
    nodes = list(graph.nodes)
    nodes_degree = np.array(list(graph.degree(nodes)))
    single_node_ids = list(np.where(nodes_degree[:, 1] == 0)[0])
    if len(single_node_ids) == 0: # no single node found
        return graph
    else:
        edges = []
        for node_id in single_node_ids:
            single_node = nodes_degree[node_id, 0]
            target_node_id = np.where(simi_matrix[single_node, nodes] == np.max(simi_matrix[single_node, nodes]))[0][0]
            target_node = nodes[target_node_id]
            edge = [single_node, target_node]
            edges.extend([edge])
        graph.add_edges_from(edges)
    
    return graph

def sort_node_degree(m_graph, nodes, node_degree):
    
    n = len(nodes)
    for i in range(0, n):
        node_degree[0, nodes[i]] = m_graph.degree(nodes[i])
    sorted_nodes = sorted(enumerate(node_degree[0, :]), key = lambda x:x[1])
    
    return sorted_nodes

def select_node(m_graph, nodes, node_degree, m_similarity, m_type, is_asc):
    
    if m_type == 0:
        # search according to node's degree
        temp_b = sort_node_degree(m_graph, nodes, node_degree)
        n = len(temp_b)
        if is_asc:
            node_id = temp_b[n-1][0]
        else:
            node_id = temp_b[0][0]
        return node_id
    else:
        node_a, node_b = np.where(m_similarity == np.max(m_similarity))
        node_a = int(node_a[0])
        node_b = int(node_b[0])
        if node_degree[0, node_a] < node_degree[0, node_b]:
            node_high = node_b
            node_low = node_a
        else:
            node_high = node_a
            node_low = node_b
        if is_asc:
            return node_high
        else:
            return node_low
        


def discover_trees(similarity, child_num, tree_level, m_type, is_asc, var_weight):
    '''
    build tree-structured clusters
    '''
    #similarity_backup = similarity.copy()
    n = similarity.shape[0]
    clusters = []
    '''
    simi_T = similarity.T
    similarity = simi_T + similarity
    for i in range(0, n):
        similarity[i, i] = 0
    '''
    similarity_backup = similarity.copy()
    
    # setting parameters
    region_flags = np.zeros([1, n])
    n_cluster = 0
    m_subgraphs = []
    clusters = []
    #var_weight = 2
    threshold = np.mean(similarity_backup) + var_weight * np.var(similarity_backup)
    
    # build the graph    
    m_graph = buildGraph(similarity, n, threshold)
    print(m_graph.number_of_edges())
    # sort nodes w.r.t degree
    region_degree = np.zeros([1, n])
    t_nodes = [x for x in range(0, n)]
    t_simi = similarity
    
    
    while np.sum(region_flags) < n:
        t_graph = nx.Graph()
        t_cluster = []
        t_region = select_node(m_graph, t_nodes, region_degree, t_simi, m_type, is_asc)
        t_cluster.extend([t_region])
        t_tag = 0
        t_len = 0
        t_idx = 1
        while len(t_cluster) != t_len:
            t_len = len(t_cluster)
            tt_tag = t_len
            for i in range(t_tag, t_len):
                # search neighbors
                new_neighbors = searchNeighbor(t_cluster[i], m_graph, child_num, similarity_backup.copy())
                # add neighbors
                for ids in new_neighbors:
                    if len(np.where(t_cluster == ids)[0]) == 0: #and similarity_backup[t_region, ids] > threshold:
                        t_cluster.extend([ids])
                        t_graph.add_edge(t_cluster[i], ids)
            t_tag = tt_tag
            t_idx = t_idx + 1
            if t_idx > tree_level:
                break
        clusters.extend([t_cluster])
        #print("clusters:")
        #print(t_cluster)
        n_cluster = n_cluster + 1
        region_flags[0, t_cluster] = 1
        m_subgraphs.extend([t_graph])
        m_graph.remove_nodes_from(t_cluster)
        t_simi[t_cluster, :] = -np.inf
        t_simi[:, t_cluster] = -np.inf
        if is_asc:
            region_degree[0, t_cluster] = -1
        else:
            region_degree[0, t_cluster] = n + 1
        # check for single nodes, then connect them to their most closed neighbor
        if len(list(m_graph.nodes)) == 0:
            break
        else:
            m_graph = checkSingleNodes(m_graph, similarity_backup.copy())
            t_nodes = list(m_graph.nodes)
            '''
            for i in range(0, len(t_nodes)):
                region_degrees[0, t_nodes[i]] = m_graph.degree(t_nodes[i])
            temp_b = sorted(enumerate(region_degrees[0, :]), key = lambda x:x[1])
            '''
    for cluster in clusters:
        if len(cluster) == 1:
            source_id = cluster[0]
            target_id = np.where(similarity_backup.copy()[source_id, :] == 
                                 np.max(similarity_backup.copy()[source_id, :]))[0]
            for i in range(0, len(clusters)):
                if target_id in clusters[i]:
                    clusters[i].extend([source_id])
                    break
            clusters.remove(cluster)
            
    return clusters, n_cluster, m_subgraphs


if __name__ == '__main__':
    
    clusters, n_cluster, m_subgraphs = discover_trees(simi_features[30].copy())
