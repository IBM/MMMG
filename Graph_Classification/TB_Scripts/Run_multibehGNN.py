#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:06:32 2022

@author: niharika.dsouza
"""
import sys
import json

# Load the configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Add the CODE_PATH to sys.path
sys.path.append(f'{config["CODE_PATH"]}/Graph_Classification/')

from utils.multigraph_object_creators import convert_to_multibehav
from utils.gnn_modules import MultiBehavioralGNN
from utils.training_helpers import  train_mG,logits_calculate_mG
from utils.evaluators import performance_evaluate

import pickle
import torch.nn.functional as F

import mlflow

# plotting tools
import matplotlib.pyplot as plt
plt.ioff()
plt.close('all')


def main(run_number, predictor,lambda_1):
    
    mlflow.set_experiment("Graph Classification Multibehavioral GNN")
    mlflow.log_param("run_number", run_number)
    
    #hyperparameters for TB dataset
    N = 3051 #number of subjects
    thresh_q = 0.9 #threshold for graph generation
    h_width = 32 #
    num_labels = 5
    
    mlflow.log_param("threshold", thresh_q)
    mlflow.log_param("h_width", h_width)

    folder_name = f'{config["SAVE_PATH"]}/Patient_Graphs/H_' + str(h_width) 
    filepath  = folder_name + '/' +str(thresh_q) + '/'+  str(run_number) + '/'
    
    filename = filepath + 'Front_end_representation.p'

    with open(filename, 'rb') as f:
           x = pickle.load(f)
    f.close()

    embeddings_concat = x['embeddings_concat']

    filename = filepath + '/Common_encoder_trained.p'

    with open(filename, 'rb') as f:
           x = pickle.load(f)
    f.close()

    embeddings_common = x['embeddings_common']
    common_model = x['models']
    outcomes = x['outcomes']
    data_splits = x['data_splits']

    #create multigraphs for multibehavioral GNN
    graphs_pat = convert_to_multibehav(embeddings_concat,filepath,h_width)
    
    
    print("Training Multibehavioral GNN  \n")
    ex_G = graphs_pat[0][0]
      
    lr = 0.001
    num_epochs = 40
   
    GCN_mb =  MultiBehavioralGNN(ex_G[0], ex_G[1], in_feats = 1, h_feats=1, num_classes =num_labels, M =h_width)
    GCN_mb =  train_mG(embeddings_concat,graphs_pat,outcomes,GCN_mb,lr,num_epochs,h_width,num_labels)
  
    logits_test_mb = F.softmax(logits_calculate_mG(graphs_pat[2],embeddings_concat[2],GCN_mb,num_labels,h_width),dim=1)
    roc_mb = performance_evaluate(outcomes[2], logits_test_mb, 'Multibehavioral GNN ', folder_name)
    
    for class_no in range(len(roc_mb)):
        mlflow.log_metric("mb_AUROC_class " + str(class_no), roc_mb[class_no])
        print("mbAUROC_class " + str(class_no) + ': ' + str(roc_mb[class_no]))
        
    #save states
    f = open(filepath + '/mb_model_'+ str(thresh_q) +'.p','wb') 
    pickle.dump(GCN_mb.state_dict(),f)
    f.close()
    
    logits_train_mb = F.softmax(logits_calculate_mG(graphs_pat[0],embeddings_concat[0],GCN_mb,num_labels,h_width),dim=1)
    logits_val_mb = F.softmax(logits_calculate_mG(graphs_pat[1],embeddings_concat[1],GCN_mb,num_labels,h_width),dim=1)
    
    logits_mb = [logits_train_mb,logits_val_mb, logits_test_mb]
    f = open(filepath + '/Multibehavioral_logits.p','wb') 
    pickle.dump(logits_mb,f)
    f.close()


if __name__ == "__main__":
    
    main(None)
    
    #ML flow logging
    
    
     