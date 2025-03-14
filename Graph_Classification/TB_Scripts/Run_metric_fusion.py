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
from utils.Metric_Network import Metric_Fusion_Network
from utils.training_helpers import  train_latentgraph
from utils.evaluators import performance_evaluate

import torch
import pickle
import torch.nn.functional as F

import mlflow,collections

# plotting tools
import matplotlib.pyplot as plt
plt.ioff()
plt.close('all')


def main(run_number, predictor,lambda_1):
    
    #ML flow logging
    
    mlflow.set_experiment("Graph Classification Metric Fusion Network")
    mlflow.log_param("run_number", run_number)
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
    
    #concatenate raw features for graph construction
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
    
    targets = [TB_dataset['outcomes'][data_splits['train_indices']],
               TB_dataset['outcomes'][data_splits['val_indices']]]    
    targets_test = TB_dataset['outcomes'][data_splits['test_indices']]
    
    feat_dict = collections.defaultdict(list)
    for i in range(len(modality_dict)):
     
        feat_dict[modality_dict[i]] = datainputs_train[modality_dict[i]].shape[1]
        
    print("Training a Metric Fusion Network ")
  
    # train latent graph learning model
    Metric_Model = Metric_Fusion_Network(32, 32, feat_dict, num_labels)
    Metric_Model.trainer(datainputs,targets,val_mask,lambda_1,batch_size=128,lr=0.0001,num_epochs=60)
  
    #evaluation
    outputs_test,logits_test = Metric_Model.forward(datainputs_test)
    logits_test = logits_test[test_mask]
 
    outputs_t,logits_t = Metric_Model.forward(datainputs[1])
    logits_val = logits_t[val_mask]
    logits_train = logits_t[~val_mask]
     
    ROC_test = performance_evaluate(targets_test, logits_test,'Metric Network Fusion',folder_name)
    
    for class_no in range(len(ROC_test)):
          mlflow.log_metric("Metric Network Fusion AUROC_class" + str(class_no), ROC_test[class_no])
      
    f = open(folder_name + '/Metric _Network_Model.p','wb') 
    pickle.dump( Metric_Model.state_dict(),f)
    f.close()
   
    logits = [logits_train,logits_val, logits_test]
    
    f = open(filepath + '/Metric _Network_logits.p','wb') 
    pickle.dump(logits,f)
    f.close()
if __name__ == "__main__":
    
    main(None)
    
    