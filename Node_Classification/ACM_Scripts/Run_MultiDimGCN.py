#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 00:21:24 2022

@author: niharika.dsouza
"""

import os, mlflow, pickle
import dgl, sys

import numpy as np
import torch,urllib,scipy

import matplotlib.pyplot as plt

from dgl.contrib.data import load_data

plt.ioff()
plt.close('all')

from utils.graph_object_creator import heterograph_creater, create_edge_list
from utils.evaluators import performance_evaluate
from utils.neural_network_modules import MultiDim_GCN
from utils.training_modules import train_GCN


if __name__ == "__main__":
    
    run_no = sys.argv[1]

    mlflow.set_experiment("ACM Dataset Experiments" )
        
    data_url = 'https://data.dgl.ai/dataset/ACM.mat'
    data_file_path = '/tmp/ACM.mat'

    urllib.request.urlretrieve(data_url, data_file_path)
    data = scipy.io.loadmat(data_file_path)

    #This creates a heterograph with a subset of the attributes and relationships
    ppA = data['PvsA'].dot(data['PvsA'].transpose())>1
    ppL = data['PvsL'].dot(data['PvsL'].transpose())>=1
    ppP = data['PvsP']
    
    #This creates a heterograph with a subset of the attributes and relationships
    G = dgl.heterograph({
        ('feat', '0', 'feat') : create_edge_list(ppA),
        ('feat', '1', 'feat') : create_edge_list(ppP), 
        ('feat', '2', 'feat') : create_edge_list(ppL)
    })

    num_rels = 3
    pvc = data['PvsC'].tocsr()
    
    c_selected = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]  
    p_selected = pvc[:, c_selected].tocoo() #tuple based 

    # old_stdout = sys.stdout

    # generate labels
    labels = pvc.indices
    labels[labels == 0] = 0
    labels[labels == 1] = 1
    labels[labels == 2] = 2
    labels[labels == 3] = 3
    labels[labels == 4] = 4
    labels[labels == 5] = 5
    # labels[labels == 6] = 6 no examples for this class
    labels[labels == 7] = 6
    labels[labels == 8] = 7
    labels[labels == 9] = 8
    labels[labels == 10] = 9
    labels[labels == 11] = 10
    labels[labels==12] =11
    labels[labels == 13] = 12
    
    labels = torch.tensor(labels).long()
    
        
    folder_name = "/Users/niharika.dsouza/Projects/refactored/ACM_Results/" +str(run_no) + '/'
    pid = p_selected.row
    shuffle = np.random.permutation(pid)
    train_idx = torch.tensor(shuffle[0:800]).long()
    val_idx = torch.tensor(shuffle[800:1000]).long()
    test_idx = torch.tensor(shuffle[1000:]).long()

    data_splits = [train_idx,val_idx,test_idx]

    
    print("Run %s of 10" %(run_no))
    mlflow.log_param("run number ", run_no)
    
    data_splits = [train_idx,val_idx,test_idx]
        
    f = open(folder_name + '/data_splits_'+ str(run_no) +'.p','wb') 
    pickle.dump(data_splits,f)
    f.close()
    
    n_classes = 13
    
    # Multi-Dimensional GCN
    print("Training Multi Dim GCN Model")
    md_model = MultiDim_GCN(G, 10 ,10, n_classes)
    train_GCN(G, md_model, data_splits, labels, 0.001,100)  #100
    Multi_DimGCN_ROC = performance_evaluate(md_model,G,labels,test_idx,'Multi Dim GCN ', folder_name)

    for class_no in range(len(Multi_DimGCN_ROC)):
            mlflow.log_metric("MultiDimGCN_AUROC_class " + str(class_no), Multi_DimGCN_ROC[class_no])

    f = open(folder_name + '/MultiDimGCN_model_'+ str(run_no) +'.p','wb') 
    pickle.dump(md_model.state_dict(),f)
    f.close()