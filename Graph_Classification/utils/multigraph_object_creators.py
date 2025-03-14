#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:15:27 2022

@author: niharika.dsouza
"""

import copy
import torch,dgl
import pickle
import collections
import scipy.sparse as spp
import numpy as np

# plotting tools
import matplotlib.pyplot as plt
plt.close('all')


def create_multiplex_graph_object(m_graph,N_types,E_types): 
    
    """
    Creates multiplex graph from heterograph objects
    
    Inputs:
        graphs_input: dgl heterograph object
    Outputs: 
        message passing object type
        g1: type I supra-adjacency
        
        (type II can be inferred from this)
    """
       
    intra_L = np.zeros((N_types*E_types,N_types*E_types))
   
    for i in range(E_types):
        
        adj = m_graph[('feat',str(i),'feat')].adjacency_matrix(transpose=True).to_dense().numpy()
        intra_L[i*N_types:i*N_types+N_types,i*N_types:i*N_types+N_types] = adj
    
    
    C = spp.coo_matrix(np.kron(np.ones((E_types,E_types)),np.eye(N_types)))
    
    intra_L  = spp.coo_matrix(intra_L)
    
    AC = intra_L.dot(C)    
    g1 = dgl.from_scipy(spp.coo_matrix(AC))

    return g1


def create_multibehav_graph_object(m_graph,N_types,E_types): 
    
    """
    Creates quotient graph and multilayered graph according to https://dl.acm.org/doi/pdf/10.1145/3340531.3412119
    
    Inputs:
        graphs_input: dgl heterograph object
    Outputs: 
        message passing object types
        g1: quotient graph 
        g2: multilayered graph
    """
     
    quotient_graph = np.zeros((N_types,N_types))
    intra_L = np.zeros((N_types*E_types,N_types*E_types))
   
    for i in range(E_types):
        
        adj = m_graph[('feat',str(i),'feat')].adjacency_matrix(transpose=True).to_dense().numpy()
        intra_L[i*N_types:i*N_types+N_types,i*N_types:i*N_types+N_types] = adj
        quotient_graph = quotient_graph + adj/E_types
    
    quotient_graph = spp.coo_matrix(quotient_graph)
    C = spp.coo_matrix(np.kron(np.ones((E_types,E_types)),np.eye(N_types))-np.eye(N_types*E_types))
    intra_L  = spp.coo_matrix(intra_L)
    mplex_g = intra_L + C 
    
    g1 = dgl.from_scipy(quotient_graph)
    g2 = dgl.from_scipy(mplex_g)
    
    return [g1,g2]

def create_mGNN_graph_object(m_graph,N_types,E_types): 
    
    """
    Creates intra and inter multigraph objects according to https://arxiv.org/pdf/2109.10119.pdf
    
    Inputs:
        graphs_input: dgl heterograph [object
    Outputs: 
        message passing object types 
        g1: inter 
        g2: intra graphs
    """
       
    intra_L = np.zeros((N_types*E_types,N_types*E_types))
   
    for i in range(E_types):
        
        adj = m_graph[('feat',str(i),'feat')].adjacency_matrix(transpose=True).to_dense().numpy()
        intra_L[i*N_types:i*N_types+N_types,i*N_types:i*N_types+N_types] = adj
    
    
    C = spp.coo_matrix(np.kron(np.ones((E_types,E_types)),np.eye(N_types))-np.eye(N_types*E_types))
    
    intra_L  = spp.coo_matrix(intra_L)
    
    g1 = dgl.from_scipy(C)
    g2 = dgl.from_scipy(intra_L)
    
    return [g1,g2]

def convert_to_multiplex(embeddings_concat,filepath,h_width,run_numberˇ):
    
    """

    Parameters
    ----------
    embeddings_concat : list of concatenated reduced modality features [train,val,test]
    filepath : path to graph objects
    h_width : width of concept space

    Returns
    -------
    graphs_pat: list of multiplex graph objects [train,val,test]

    """
    
    G_train = [] 
    G_val = []
    G_test = []
    
    N_types = embeddings_concat[0].shape[1]
    E_types = h_width
    
    print("Creating Multiplex Graphs")
    
    for pat_no in range(embeddings_concat[0].shape[0]):

            with open(filepath + '/Train/' + str(pat_no) + '.p','rb') as f:              
               graph_output = pickle.load(f)

            f.close()
            
            mG = create_multiplex_graph_object(graph_output,N_types,E_types)   
            G_train.append(mG)
            
            print("Patient " + str(pat_no)+ " processed")
          
    
    for pat_no  in range(embeddings_concat[1].shape[0]):

            with open(filepath + '/Val/' + str(pat_no) + '.p','rb') as f:
               graph_output = pickle.load(f)
               
            f.close()
            
            mG = create_multiplex_graph_object(graph_output,N_types,E_types)        
            G_val.append(mG)
            
            print("Patient " +str(pat_no)+ " processed")

            
            
    for pat_no  in range(embeddings_concat[2].shape[0]):

            with open(filepath + '/Test/' + str(pat_no) + '.p','rb') as f:
               graph_output = pickle.load(f)

            f.close()
            
            mG = create_multiplex_graph_object(graph_output,N_types,E_types)        
            G_test.append(mG)  
            
            print("Patient " +str(pat_no)+ " processed")

            
    graphs_pat = [G_train,G_val,G_test]
   
    return graphs_pat

def convert_to_multibehav(embeddings_concat,filepath,h_width):
    
    """

    Parameters
    ----------
    embeddings_concat : list of concatenated reduced modality features [train,val,test]
    filepath : path to graph objects
    h_width : width of concept space

    Returns
    -------
    graphs_pat: list of multibehav graph objects [train,val,test]

    """
    
    G_train = [] 
    G_val = []
    G_test = []
    
    N_types = embeddings_concat[0].shape[1]
    E_types = h_width
    
    print("Creating Multibehavioral GNN Graphs")
    
    for pat_no in range(embeddings_concat[0].shape[0]):

            with open(filepath + '/Train/' + str(pat_no) + '.p','rb') as f:              
               graph_output = pickle.load(f)

            f.close()
            
            mG = create_multibehav_graph_object(graph_output,N_types,E_types)   
            G_train.append(mG)
            
            print("Patient " + str(pat_no)+ " processed")
          
    
    for pat_no  in range(embeddings_concat[1].shape[0]):

            with open(filepath + '/Val/' + str(pat_no) + '.p','rb') as f:
               graph_output = pickle.load(f)
               
            f.close()
            
            mG = create_multibehav_graph_object(graph_output,N_types,E_types)        
            G_val.append(mG)
            
            print("Patient " +str(pat_no)+ " processed")

            
            
    for pat_no  in range(embeddings_concat[2].shape[0]):

            with open(filepath + '/Test/' + str(pat_no) + '.p','rb') as f:
               graph_output = pickle.load(f)

            f.close()
            
            mG = create_multibehav_graph_object(graph_output,N_types,E_types)        
            G_test.append(mG)  
            
            print("Patient " +str(pat_no)+ " processed")

            
    graphs_pat = [G_train,G_val,G_test]
   
    return graphs_pat

def convert_to_graphs(embeddings_concat,filepath,h_width):
    
    """

    Parameters
    ----------
    embeddings_concat : list of concatenated reduced modality features [train,val,test]
    filepath : path to graph objects
    h_width : width of concept space

    Returns
    -------
    graphs_pat: list of graph objects [train,val,test]

    """
    
    G_train = [] 
    G_val = []
    G_test = []
    
    for pat_no  in range(embeddings_concat[0].shape[0]):

            with open(filepath + '/Train/' + str(pat_no) + '.p','rb') as f:
                G_train.append(pickle.load(f))

            f.close()
            print("Patient " +str(pat_no)+ " processed")
            
    
    for pat_no  in range(embeddings_concat[1].shape[0]):

            with open(filepath + '/Val/' + str(pat_no) + '.p','rb') as f:
                G_val.append(pickle.load(f))

            f.close()
            print("Patient " +str(pat_no)+ " processed")

            
            
    for pat_no  in range(embeddings_concat[2].shape[0]):

            with open(filepath + '/Test/' + str(pat_no) + '.p','rb') as f:
                G_test.append(pickle.load(f))

            f.close()
            print("Patient " +str(pat_no)+ " processed")
            
            
    graphs_pat = [G_train,G_val,G_test]
    
    return graphs_pat

def convert_to_mGNN(embeddings_concat,filepath,h_width):
    
    """

    Parameters
    ----------
    embeddings_concat : list of concatenated reduced modality features [train,val,test]
    filepath : path to graph objects
    h_width : width of concept space

    Returns
    -------
    graphs_pat: list of mGNN graph objects [train,val,test]

    """
    
    G_train = [] 
    G_val = []
    G_test = []
    
    N_types = embeddings_concat[0].shape[1]
    E_types = h_width
    
    print("Creating Multibehavioral GNN Graphs")
    
    for pat_no in range(embeddings_concat[0].shape[0]):

            with open(filepath + '/Train/' + str(pat_no) + '.p','rb') as f:              
               graph_output = pickle.load(f)

            f.close()
            
            mG = create_mGNN_graph_object(graph_output,N_types,E_types)   
            G_train.append(mG)
            
            print("Patient " + str(pat_no)+ " processed")
          
    
    for pat_no  in range(embeddings_concat[1].shape[0]):

            with open(filepath + '/Val/' + str(pat_no) + '.p','rb') as f:
               graph_output = pickle.load(f)
               
            f.close()
            
            mG = create_mGNN_graph_object(graph_output,N_types,E_types)        
            G_val.append(mG)
            
            print("Patient " +str(pat_no)+ " processed")

            
            
    for pat_no  in range(embeddings_concat[2].shape[0]):

            with open(filepath + '/Test/' + str(pat_no) + '.p','rb') as f:
               graph_output = pickle.load(f)

            f.close()
            
            mG = create_mGNN_graph_object(graph_output,N_types,E_types)        
            G_test.append(mG)  
            
            print("Patient " +str(pat_no)+ " processed")

            
    graphs_pat = [G_train,G_val,G_test]
   
    return graphs_pat

def convert_to_feature_graph(features):
    
    """
    
    Create unimodal graphs from the reduced concatenated features for GCN
    
    Inputs:
        features: concatenated features
    Outputs: 
        graphs_pat: unimodal graphs
    
    """
    
    graphs_pat =[]
    
    for i in range(len(features)):
        
        g_mat = []
        
        for pat_no in range(features[i].size()[0]):
            
            v_emb = features[i][pat_no,:].reshape(-1,1)
            thresh = torch.quantile(v_emb,0.8)
            
            v = (v_emb>thresh).float()
            adj = spp.coo_matrix((v.mm(v.transpose(0, 1))).mul(torch.ones(v.shape[0])- torch.eye(v.shape[0])))
            
            g = dgl.from_scipy(adj)

            g_mat.append(g)

            print("Patient %d processed" % pat_no)
            
        graphs_pat.append(g_mat)
        
    return graphs_pat

def create_mmfeature_graph(features,list_sizes,node_types,thresh_q):
    
    """
    
    Create multimodal graphs from the reduced concatenated features for RGCN
    
    Inputs:
        features: concatenated features [train,val,test]
        list_sizes: list of feature sizes
    Outputs: 
        graphs_pat: multi-modal graphs [train,val,test]
    
    """

    dict_nodes = collections.defaultdict()
    
    for k,v in enumerate(list_sizes):
        dict_nodes[node_types[k]] = v

    graphs_pat =[]
    
    for i in range(len(features)):
        
        g_mat = []
        
        for pat_no in range(features[i].size()[0]):
            
            fs = 0
            m_g = collections.defaultdict()
            n_edges = {}
        
            for m,feat_s in enumerate(list_sizes):

                # create adjacency matrix
                feat_mod = features[i][pat_no][fs:fs+feat_s]
                
                thresh = torch.quantile(feat_mod,thresh_q)
                rank_one_rep = (feat_mod>thresh).float()
        
                v = rank_one_rep.reshape((-1, 1))
                adj = spp.coo_matrix((v.mm(v.transpose(0, 1))).mul(torch.ones(v.shape[0])- torch.eye(v.shape[0])))
        
                rows, cols = adj.nonzero()
            
                n_edges[m] = adj.sum()
                list_edge_tuples = []
                
                for ind, r in enumerate(rows):
                    list_edge_tuples.append((r, cols[ind]))

                # create graph from adjacency matrix, then add to heterograph

                if any(v):  # filter for empty graphs
                    m_g[(node_types[m], str(m), node_types[m])] = list_edge_tuples
        
                fs = fs+feat_s
        
            # create multigraph object
            multigraph = dgl.heterograph(m_g, dict_nodes)   
            g_mat.append(multigraph)
        
            print("Patient %d processed" % pat_no)
        
        graphs_pat.append(g_mat)
        