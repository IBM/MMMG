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
from utils.transformer_utils import feature_transformer
from utils.training_helpers import  train_latentgraph
from utils.evaluators import performance_evaluate

import torch
import pickle
import torch.nn.functional as F
import numpy as np

import mlflow

# plotting tools
import matplotlib.pyplot as plt
plt.ioff()
plt.close('all')

def main(run_number, predictor,lambda_1):
    
    #ML flow logging
    
    mlflow.set_experiment("Graph Classification Transformer")
    mlflow.log_param("run_number", run_number)
    
  
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
    targets = x['outcomes']
    
    filename = filepath + '/Front_end_representation.p'

    with open(filename, 'rb') as f:
           x = pickle.load(f)
    f.close()
    
    dataset = x['embeddings_concat']
    in_feats = dataset[0].shape[1]
    [datainputs_train,datainputs_val,datainputs_test] =  [dataset[0],dataset[1],dataset[2]]
    
    print("Training a transformer model ")
   
    # train transformer
    trans_model = feature_transformer(in_feats, num_heads=1, h_feats= 256, num_classes= num_labels)
    
    trans_model.trainer(dataset, targets, lr =0.001,num_epochs = 10)
    
    logits_test = trans_model.forward(datainputs_test)
    targets_test = targets[2]
     
    ROC_test = performance_evaluate(targets_test, logits_test,'Transformer',folder_name)
    
    for class_no in range(len(ROC_test)):
          mlflow.log_metric("Transformer AUROC_class" + str(class_no), ROC_test[class_no])
      
    f = open(folder_name + '/Transformer.p','wb') 
    pickle.dump(trans_model.state_dict(),f)
    f.close()
    
    logits_train = trans_model(datainputs_train)
    logits_val = trans_model(datainputs_val)
    logits_test = trans_model(datainputs_test)
   
    logits = [logits_train,logits_val, logits_test]
    
    f = open(filepath + '/Transformer_logits.p','wb') 
    pickle.dump(logits,f)
    f.close()

if __name__ == "__main__":
    
   
    main(None)
    