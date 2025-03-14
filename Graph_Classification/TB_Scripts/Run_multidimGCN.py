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

from utils.multigraph_object_creators import convert_to_graphs
from utils.gnn_modules import MultiDimGCN_stack
from utils.training_helpers import  train_gcn,logits_calculate
from utils.evaluators import performance_evaluate

import pickle
import torch.nn.functional as F

import mlflow

# plotting tools
import matplotlib.pyplot as plt
plt.ioff()
plt.close('all')

def main(run_number, predictor,lambda_1):
    
    mlflow.set_experiment("Graph Classification MultiDim GCN")
    mlflow.log_param("run_number", run_number)
    
    #load modality specific features from file
    
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
         
    #create multigraphs
    graphs_pat = convert_to_graphs(embeddings_concat,filepath,h_width)
    
    
    print("Training MultiDimensional GCN network \n")
    ex_G = graphs_pat[0][0]
    ex_G.features = embeddings_concat[0][0].view(-1,1)
      
    lr = 0.005
    num_epochs = 40
   
    MGCN_model = MultiDimGCN_stack(ex_G, in_size = 1, hidden_size = 1 , out_size = 1, num_classes= num_labels, alpha = 0.5)
    MGCN_model = train_gcn(embeddings_concat,graphs_pat,outcomes, MGCN_model,lr,num_epochs,num_labels)
    
    logits_test_MGCN = F.softmax(logits_calculate(graphs_pat[2],embeddings_concat[2],MGCN_model,num_labels),dim=1)
    roc_MGCN = performance_evaluate(outcomes[2], logits_test_MGCN, 'Multi Dimensional GCN ', folder_name)
     
    for class_no in range(len(roc_MGCN)):
        mlflow.log_metric("MGCN_AUROC_class " + str(class_no), roc_MGCN[class_no])
        print("MGCN_AUROC_class " + str(class_no) + ': ' + str(roc_MGCN[class_no]))
        
    #save states
    f = open(filepath + '/MGCN_model_'+ str(thresh_q) +'.p','wb') 
    pickle.dump(MGCN_model.state_dict(),f)
    f.close()
    
    logits_train_MGCN = F.softmax(logits_calculate(graphs_pat[0],embeddings_concat[0],MGCN_model,num_labels),dim=1)
    logits_val_MGCN = F.softmax(logits_calculate(graphs_pat[1],embeddings_concat[1],MGCN_model,num_labels),dim=1)
    
    logits_MGCN = [logits_train_MGCN,logits_val_MGCN, logits_test_MGCN]
    f = open(filepath + '/MGCN_logits.p','wb') 
    pickle.dump(logits_MGCN,f)
    f.close()

if __name__ == "__main__":
    
    main(None)
    
    #ML flow logging
    
    
    

