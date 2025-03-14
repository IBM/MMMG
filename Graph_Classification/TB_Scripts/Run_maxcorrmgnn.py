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

from utils.data_prep import prepare_dataset
from utils.multigraph_object_creators import convert_to_multiplex
from utils.HGR_modules import HGR_Network
from utils.training_helpers import  train_latentgraph
from utils.evaluators import performance_evaluate

import torch
import pickle,collections
import torch.nn.functional as F

import mlflow

# plotting tools
import matplotlib.pyplot as plt
plt.ioff()
plt.close('all')

def main(run_number, predictor,lambda_1):
    
    mlflow.set_experiment("Node Classification MaxCorrMGNN")
    mlflow.log_param("run_number", run_number)
    mlflow.log_param("predictor", predictor)
    mlflow.log_param("loss tradeoff", lambda_1)
  
    #hyperparameters for TB dataset
    N = 3051 #number of subjects
    num_labels = 5
    h_width = 32
    thresh_q = 0.9
    
    folder_name = f'{config["SAVE_PATH"]}/Patient_Graphs/H_' + str(h_width) 
    filepath  = folder_name + '/' +str(thresh_q) + '/'+  str(run_number) + '/'

    filename = filepath + '/Common_encoder_trained.p'

    with open(filename, 'rb') as f:
           x = pickle.load(f)
    f.close()

    #pick up data splits
    data_splits = x['data_splits']
    train_s = len(data_splits['train_indices'])
    val_s = len(data_splits['val_indices'])
    test_s = len(data_splits['test_indices'])

    #pick up raw data
    with open(f'{config["CODE_PATH"]}/TB_data.pickle','rb') as f:
         TB_dataset = pickle.load(f)
    f.close()
    [dataset,mask,models,modality_dict,params] = prepare_dataset(TB_dataset, data_splits)
    
    #setup data for inductive learning
    datainputs_train = collections.defaultdict(list)
    datainputs_val = collections.defaultdict(list)
    datainputs_test = collections.defaultdict(list)
    
    for k,v in enumerate(dataset):
        datainputs_train[v] = dataset[v][0]
        datainputs_val[v] = torch.cat((dataset[v][0],dataset[v][1]))
        datainputs_test[v] = torch.cat((dataset[v][0],dataset[v][2]))
        
    datainputs = [datainputs_train,datainputs_val]
    val_mask = torch.cat((torch.zeros((train_s,1)), torch.ones((val_s,1))),dim=0) ==1
    val_mask = val_mask[:,0]
    test_mask = torch.cat((torch.zeros((train_s,1)), torch.ones((test_s,1))),dim=0) ==1
    test_mask = test_mask[:,0]
  
    #setup targets for inductive learning
    targets = [TB_dataset['outcomes'][data_splits['train_indices']],
               TB_dataset['outcomes'][data_splits['val_indices']]]
    targets_test = TB_dataset['outcomes'][data_splits['test_indices']]
    
    #
    feat_dict = collections.defaultdict(list)
    for i in range(len(modality_dict)):
     
        feat_dict[modality_dict[i]] = datainputs_train[modality_dict[i]].shape[1]
    
    print("Training the Max Corr MGNN ")
   
    HGR_Model = HGR_Network(in_feats=64, h_feats=64, modality_dict = feat_dict, num_classes= num_labels,drop_rate=0.7)
    
   # first train only with HGR for 50 epochs to initialise model from reasonable starting point
   # use large batch size to ensure that sample covariance estimates are not too far away from true covariance estimates
    HGR_Model.trainer(datainputs,targets,val_mask,lambda_1 = 1.0,batch_size=1024,lr=0.001,num_epochs=50, pred_flag=predictor)
    
    #fine tune with soft HGR as regulariser for the graph construction
    HGR_Model.trainer(datainputs,targets,val_mask,lambda_1,batch_size=128,lr=0.0001,num_epochs=40, pred_flag=predictor)
            
  
    #evaluation
    if predictor == 'GNN':
        logits_test = HGR_Model.gnn_forward(HGR_Model.HGR_forward(datainputs_test)[1])
        logits_train = HGR_Model.gnn_forward(HGR_Model.HGR_forward(datainputs_train)[1])
        logits_val = HGR_Model.gnn_forward(HGR_Model.HGR_forward(datainputs[1])[1])[val_mask]
    else:
        logits_test = HGR_Model.mlp_forward(HGR_Model.HGR_forward(datainputs_test)[0])
        logits_train = HGR_Model.mlp_forward(HGR_Model.HGR_forward(datainputs_train)[0])
        logits_val = HGR_Model.mlp_forward(HGR_Model.HGR_forward(datainputs[1])[0])[val_mask]
    
    logits_test = logits_test[test_mask]
    
    ROC_test = performance_evaluate(targets_test, logits_test,'Max Corr MGNN',folder_name)
    
    for class_no in range(len(ROC_test)):
          mlflow.log_metric("Max Corr MGNN AUROC_class" + str(class_no), ROC_test[class_no])
      
    f = open(folder_name + '/MaxCorrMGNN_model.p','wb') 
    pickle.dump(HGR_Model.state_dict(),f)
    f.close()
    
    logits = [logits_train,logits_val, logits_test]
    
    f = open(filepath + '/MaxCorrMGNN_logits.p','wb') 
    pickle.dump(logits,f)
    f.close()
    
if __name__ == "__main__":
    
    ## descriptions
#    run_number = seed for fold
#    predictor =  #GNN predictor for maxcorrmgnn, change to MLP for baseline https://ojs.aaai.org/index.php/AAAI/article/view/4464
#    lambda_1 =  #tradeoff between HGR and cross entropy loss, 0.01 default, 
# setting for sequential training of projector and GNN lambda = 0.0
   
    main()
 