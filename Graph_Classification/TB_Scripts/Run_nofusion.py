#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 00:57:40 2022

@author: niharika.dsouza
"""

import sys
import json

# Load the configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Add the CODE_PATH to sys.path
sys.path.append(f'{config["CODE_PATH"]}/Graph_Classification/')

from utils.data_prep import prepare_modality_nofusion
from utils.training_helpers import  train_modality_predictors,logits_calculate
from utils.evaluators import performance_evaluate

import pickle
import torch.nn.functional as F

import mlflow

# plotting tools
import matplotlib.pyplot as plt
plt.ioff()
plt.close('all')

def main(run_number, predictor,lambda_1):
    
    mlflow.set_experiment("Graph Classification: No fusion baseline")
    mlflow.log_param("run_number", run_number)
    
    #load modality specific features from file
    
    with open(f'{config["CODE_PATH"]}/TB_data.pickle','rb') as f:
         TB_dataset = pickle.load(f)
    f.close()
    
    #splits for training/test/val
  
    N = 3051 #number of subjects
    thresh_q = 0.9 #threshold for graph generation
    h_width = 32 #width of concept space
    num_labels = 5
    
    mlflow.log_param("threshold", thresh_q)
    mlflow.log_param("h_width", h_width)

    folder_name = f'{config["SAVE_PATH"]}/Patient_Graphs/H_' + str(h_width) + '/' + str(thresh_q) + '/' +str(run_number) + '/'

    filename = folder_name + '/Common_encoder_trained.p'

    with open(filename, 'rb') as f:
           x = pickle.load(f)
    f.close()

    embeddings_common = x['embeddings_common']
    outcomes = x['outcomes']
    data_splits = x['data_splits']

    emb_dim = [8,128,64,128,64,4] 
    print("Train individual modality predictors")
   
    
    [dataset,models,modality_dict,params] = prepare_modality_nofusion(TB_dataset,data_splits,width=400)
    
    [logits_all,models,roc] = train_modality_predictors(dataset,models,params,outcomes,folder_name)
   
    for key,val in enumerate(roc):
        
        roc_mod = roc[val]
        
        for class_no in range(len(roc_mod)):
             mlflow.log_metric("Modality_" + val +  "_AUROC_class " + str(class_no), roc_mod[class_no])
   
    dict_save = {'logits_train':logits_all[0], 'logits_val':logits_all[1], 'logits_test': logits_all[2], 'outcomes': outcomes }
   
    f = open(folder_name + '/modalitypredictors_logits_and_outcomes.p','wb') 
    pickle.dump(dict_save,f)
    f.close()
    
    

if __name__ == "__main__":
    
    main(None)
    
   