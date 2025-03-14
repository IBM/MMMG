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

from utils.multigraph_object_creators import convert_to_graphs,create_mmfeature_graph
from utils.gnn_modules import Relational_GCN
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

    #ML flow logging
    
    mlflow.set_experiment("Graph Classification RGCN ")
    mlflow.log_param("run_number", run_number)
    mlflow.log_param("predictor", predictor)
  
    #hyperparameters for TB dataset
    N = 3051 #number of subjects
    thresh_q = 0.9 #threshold for graph generation
    h_width = 32 #
    num_labels = 5

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

    #use created multigraphs
    if predictor == 'multiplex-like-graphs':
        graphs_pat = convert_to_graphs(embeddings_concat,filepath,h_width)
    
    ## This block runs RGCN variant on the reduced modality feature graphs defining planar connectivity  
    
    elif predictor == 'reduced-modality-hetero-graphs':
        list_sizes = [8,128,64,128,64,4]
        node_types = ['demographic_feat','ct_feat','snp_feat','clinical_feat','regimen_feat','cont_feat']
        graphs_pat = create_mmfeature_graph(embeddings_concat,list_sizes,node_types,0)
    ##
    
    else:
        print('Graph specification type unsupported')
    
    print("Training R-GCN network \n")
    ex_G = graphs_pat[0][0]
    ex_G.features = embeddings_concat[0][0].view(-1,1)
      
    lr = 0.001 
    num_epochs = 40
   
    if predictor == 'multiplex-like-graphs':
        RGCN_model = Relational_GCN(ex_G, in_size = 1, hidden_size = 1, num_classes= num_labels)
    
    ### Comment line 84, uncomment the following to run RGCN variant on the reduced modality feature graphs defining planar connectivity 
    elif predictor == 'reduced-modality-hetero-graphs':
        RGCN_model = Relational_GCN(ex_G, in_size = 1, hidden_size = 1, num_classes= num_labels, M = 6)
    
    else:
        print("Incorrect specification of predictor, please check usage and try again")
    ###
    
    RGCN_model = train_gcn(embeddings_concat,graphs_pat,outcomes,RGCN_model,lr,num_epochs,num_labels)
    
    logits_test_RGCN = F.softmax(logits_calculate(graphs_pat[2],embeddings_concat[2],RGCN_model,num_labels),dim=1)
    roc_RGCN = performance_evaluate(outcomes[2], logits_test_RGCN, 'Relational GCN ', folder_name)
    
    for class_no in range(len(roc_RGCN)):
        mlflow.log_metric("RGCN_AUROC_class " + str(class_no), roc_RGCN[class_no])
        print("RGCN_AUROC_class " + str(class_no) + ': ' + str(roc_RGCN[class_no]))
        
    #save states
    f = open(filepath + '/RGCN_model_'+ str(thresh_q) +'.p','wb') 
    pickle.dump(RGCN_model.state_dict(),f)
    f.close()
    
    logits_train_RGCN = F.softmax(logits_calculate(graphs_pat[0],embeddings_concat[0],RGCN_model,num_labels),dim=1)
    logits_val_RGCN = F.softmax(logits_calculate(graphs_pat[1],embeddings_concat[1],RGCN_model,num_labels),dim=1)
    
    logits_RGCN = [logits_train_RGCN,logits_val_RGCN, logits_test_RGCN]
    f = open(filepath + '/RGCN_logits.p','wb') 
    pickle.dump(logits_RGCN,f)
    f.close()

if __name__ == "__main__":
    
    main(None)
    
    