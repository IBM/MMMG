#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 23:54:20 2022

@author: niharika.dsouza
"""

import os, mlflow, pickle
import dgl, sys

import numpy as np
import torch

import matplotlib.pyplot as plt

from dgl.contrib.data import load_data

plt.ioff()
plt.close('all')

from utils.graph_object_creator import heterograph_creater, create_edge_list
from utils.performance_evaluate import performance_evaluate_multiplex, performance_evaluate
from utils.neural_network_modules import MultiplexSGCN
from utils.training_modules import train_multiplex
from utils.evaluators import  performance_evaluate_multiplex
from utils.graph_object_creator import create_CIKM_prop_graph, create_mGNN_prop_graph, create_multiplex_prop_graph

if __name__ == "__main__":
    
    run_no = sys.argv[1]
    
    data = load_data(dataset='aifb')
    mlflow.set_experiment("AIFB Dataset Experiments " )

    num_nodes = data.num_nodes
    num_classes = data.num_classes

    labels = data.labels

    def_train_idx = data.train_idx
    def_test_idx = data.test_idx
    
    labeled_indices = np.concatenate((def_train_idx,def_test_idx),axis=0)
    
    # split training and validation set
    num_label = len(labeled_indices) 
    ptation = np.random.permutation(num_label)
    val_idx = labeled_indices[ ptation[:int(np.ceil(0.1*(num_label)))]]
    test_idx = labeled_indices[ptation[int(np.ceil(0.1*(num_label))): int(np.ceil(0.3*(num_label)))]]
    train_idx = labeled_indices[ptation[int(np.ceil(0.3*(num_label))):]]

    
    edge_type = torch.from_numpy(data.edge_type)
    
    num_rels = len(torch.unique(edge_type)) #count edge types with non-emply relations
    labels = torch.from_numpy(labels).view(-1)

    # create graph
    G = heterograph_creater(data)
    
    folder_name = "/Users/niharika.dsouza/Projects/refactored/AIFB_Results/" +str(run_no) + '/'
    if not (os.path.exists(folder_name)):os.makedirs(folder_name)
    
    print("Run %s of 10" %(run_no))
    mlflow.log_param("run number ", run_no)
    
    data_splits = [train_idx,val_idx,test_idx]
        
    f = open(folder_name + '/data_splits_'+ str(run_no) +'.p','wb') 
    pickle.dump(data_splits,f)
    f.close()
    
    n_classes = 4
    
    print("Training sGCN Multiplex Model")
    
    G_mplex = create_multiplex_prop_graph(G)
    
    model_multiplex_sGCN = MultiplexSGCN(G_mplex, 8, 8, num_classes, num_rels)
    train_multiplex(G_mplex[0], G_mplex[1], model_multiplex_sGCN,data_splits, labels, num_levels= num_rels, lr = 0.002, num_epochs = 40)
    mplex_sGCN_ROC = performance_evaluate_multiplex(model_multiplex_sGCN,G_mplex,labels,test_idx,num_rels,'Multiplex SGCN ', folder_name)


    for class_no in range(len(mplex_sGCN_ROC)):
              mlflow.log_metric("mplex_sG_AUROC_class " + str(class_no), mplex_sGCN_ROC[class_no])
    
    f = open(folder_name + '/Mplex_model_'+ str(run_no) +'.p','wb') 
    pickle.dump(model_multiplex_sGCN.state_dict(),f)
    f.close()
        
    