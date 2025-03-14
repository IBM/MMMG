#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 00:11:25 2022

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

from utils.graph_object_creator import heterograph_creater, create_mGNN_prop_graph
from utils.evaluators import performance_evaluate_multiplex
from utils.neural_network_modules import mGNN
from utils.training_modules import train_multiplex


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
    
    print("Training mGNN Model")
    G_mGNN = create_mGNN_prop_graph(G)
    mGNN_model = mGNN(G_mGNN, 8, 8, 1, n_classes, num_rels)
    train_multiplex(G_mGNN[0],G_mGNN[1], mGNN_model,data_splits, labels, num_levels =num_rels, lr = 0.001, num_epochs = 50) #30
    mGNN_ROC = performance_evaluate_multiplex(mGNN_model,G_mGNN,labels,test_idx,num_rels,'mGNN ',folder_name)
   
    for class_no in range(len(mGNN_ROC)):
        mlflow.log_metric("mGNN_AUROC_class " + str(class_no), mGNN_ROC[class_no])
    
    f = open(folder_name + '/mGNN_model_'+ str(run_no) +'.p','wb') 
    pickle.dump(mGNN_model.state_dict(),f)
    f.close()