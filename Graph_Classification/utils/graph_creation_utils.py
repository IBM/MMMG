#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:43:14 2022

@author: niharika.dsouza
"""
import sys
import json

# Load the configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Add the CODE_PATH to sys.path
sys.path.append(config['CODE_PATH'])

from utils.autoencoder_modules import AE_Model

import copy
import torch,dgl
import pickle
import collections
import scipy.sparse as spp

# plotting tools
import matplotlib.pyplot as plt
plt.close('all')


def multigraph_object_creator(common_model,common_encodings,modality_inputs,thresh_q):
    
    """
    Creates a multigraph from the endcoder and embeddings
    
    Inputs: 
        common_model : Common Autoencoder 
        modality_inputs: concatenated embeddings used to train the common_model per patient
        common_encodings: hidden representation for all patients from common autoencoder per patient
        thresh_q: quantile for thresholding
        
    Outputs:
        multi_graphs: multigraph dgl object
        diff_graph_gen: threshold object for graph generation
        
    """

    M = common_encodings.size()[0] #size of encoding - no of multiplex layers in patient graph
    P = modality_inputs.size()[0] #size of graph - no of nodes
    
    diff_graph_gen = torch.zeros([P,M]) #initalise difference matrix
    

    for p in range(P): 
            
        #compute feature importances
        input_per = copy.deepcopy(modality_inputs) #prevents accidental override
        input_per[p] = 0.0
        hidd_per = common_model(input_per)[1]
        diff_graph_gen[p] = abs(hidd_per - common_encodings)
    
    #set threshold for each row based on quantile
 
    thresh = torch.quantile(diff_graph_gen,thresh_q,dim=0).expand_as(diff_graph_gen)
    

    rank_one_rep = (diff_graph_gen > thresh).float() #store active encodings for each latent dimension
    
    g = collections.defaultdict()
    
    n_edges = {}
    
    for m in range(M):

       # create planar adjacency matrix
        v = rank_one_rep[:, m].reshape((-1, 1))
       
       #removes self-connections
        adj = spp.coo_matrix((v.mm(v.transpose(0, 1))).mul(torch.ones(v.shape[0])- torch.eye(v.shape[0])))

        rows, cols = adj.nonzero()

        n_edges[m] = adj.sum()
        list_edge_tuples = []
        for ind, r in enumerate(rows):
            list_edge_tuples.append((r, cols[ind]))

        # create planar graph from adjacency matrix, then add to heterograph
        if any(v):  # filter for empty graphs
            g[("feat", str(m), "feat")] = list_edge_tuples

    # create multigraph object, second argument ensures number of nodes are consistent across planes
    multi_graph = dgl.heterograph(g, {"feat": P})   
    
    return multi_graph,diff_graph_gen,rank_one_rep

def convert_to_multigraphs(embeddings_concat,embeddings_common,common_model,thresh_q,folder_name,run_number):
   
    """

    Build and save multigraph objects using concept space autoencoder
    ----------
    embeddings_concat : list of concatenated features [train,val,test]
    embeddings_common : list of encoded features through concept space [train,val,test]
    common_model : concept space autoencoder model
    thresh_q : threshold for graph generation
    folder_name : folder_name for storage
    
    Returns
    -------
    None.

    """
    
    for pat_no in range(embeddings_concat[0].size()[0]):
        
        graph_output = multigraph_object_creator(common_model,embeddings_common[0][pat_no], embeddings_concat[0][pat_no], thresh_q)
        
        f = open(folder_name+ '/' +str(thresh_q) + '/' + str(run_number) + '/Train/' + str(pat_no) + '.p','wb')
        pickle.dump(graph_output[0],f)
        
        f.close()

        print('\n Patient ' + str(pat_no) + ' processed')
        
        
    for pat_no in range(embeddings_concat[1].size()[0]):
        
        graph_output = multigraph_object_creator(common_model,embeddings_common[1][pat_no], embeddings_concat[1][pat_no], thresh_q)
        
        f = open(folder_name + '/' + str(thresh_q) + '/' + str(run_number) + '/Val/' + str(pat_no) + '.p','wb')
        pickle.dump(graph_output[0],f)
        
        f.close()
        
        print('\n Patient ' + str(pat_no) + ' processed')
        
    for pat_no in range(embeddings_concat[2].size()[0]):
        

        graph_output = multigraph_object_creator(common_model,embeddings_common[2][pat_no], embeddings_concat[2][pat_no], thresh_q)
        
        f = open(folder_name + '/' + str(thresh_q) + '/' + str(run_number) + '/Test/' + str(pat_no) + '.p','wb')
        pickle.dump(graph_output[0],f)
        f.close()
        
        print('\n Patient ' + str(pat_no) + ' processed')