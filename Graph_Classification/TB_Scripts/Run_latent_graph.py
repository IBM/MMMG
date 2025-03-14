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
from utils.gnn_modules import DGraph_GAT
from utils.training_helpers import  train_latentgraph
from utils.evaluators import performance_evaluate

import torch
import pickle
import torch.nn.functional as F

import mlflow

# plotting tools
import matplotlib.pyplot as plt
plt.ioff()
plt.close('all')

def main(run_number, predictor,lambda_1):
    
    #ML flow logging
    
    mlflow.set_experiment("Graph Classification Differentiable Graph Module")
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
    train_s = len(data_splits['train_indices'])
    val_s = len(data_splits['val_indices'])
    test_s = len(data_splits['test_indices'])

    #pick up raw data
    with open(f'{config["CODE_PATH"]}/TB_data.pickle','rb') as f:
         TB_dataset = pickle.load(f)
    f.close()
    [dataset,mask,models,modality_dict,params] = prepare_dataset(TB_dataset, data_splits)
    
    #concatenate raw features for graph construction
    ds_concat_t = torch.zeros((dataset[list(dataset.keys())[0]][0].shape[0],1))
    ds_concat_v = torch.zeros((dataset[list(dataset.keys())[0]][1].shape[0],1))
    ds_concat_te = torch.zeros((dataset[list(dataset.keys())[0]][2].shape[0],1))
    
    for k,v in enumerate(dataset):
        ds_concat_t = torch.cat((ds_concat_t,dataset[v][0]),dim=1)
        ds_concat_v = torch.cat((ds_concat_v,dataset[v][1]),dim=1)
        ds_concat_te = torch.cat((ds_concat_te,dataset[v][2]),dim=1)
       
    dataset = [ds_concat_t[:,1:],ds_concat_v[:,1:],ds_concat_te[:,1:]]
    datainputs_train = dataset[0]
    datainputs_val = torch.cat((dataset[0],dataset[1]))
    datainputs_test = torch.cat((dataset[0],dataset[2]))
     
    in_feats = dataset[0].shape[1]
    
    datainputs = [datainputs_train,datainputs_val]
    
    # objects to set up inductive graph reasoning, masks are used to hide test/val data during training
    val_mask = torch.cat((torch.zeros((train_s,1)), torch.ones((val_s,1))),dim=0) ==1
    val_mask = val_mask[:,0]
    test_mask = torch.cat((torch.zeros((train_s,1)), torch.ones((test_s,1))),dim=0) ==1
    test_mask = test_mask[:,0]
    
    targets = [TB_dataset['outcomes'][data_splits['train_indices']],
               TB_dataset['outcomes'][data_splits['val_indices']]]
    targets_test = TB_dataset['outcomes'][data_splits['test_indices']]
    
    print("Training a GAT with differentiable graph module ")
      
    lr = 0.001 
    num_epochs = 40
   
    # train latent graph learning model
    Dgraph_model = DGraph_GAT(in_feats, h_feats=256, num_classes=num_labels)
    Dgraph_model = train_latentgraph(Dgraph_model,datainputs,targets,val_mask,batch_size=2135,lr=0.01,num_epochs=600) #they perform full batch grad. descent
  
    #evaluation
    logits_test = Dgraph_model.forward(datainputs_test)
    logits_test = logits_test[test_mask]
    
    pred_test = logits_test.argmax(1)
    
     
    ROC_test = performance_evaluate(targets_test, logits_test,'DGraph GAT',folder_name)
    
    for class_no in range(len(ROC_test)):
          mlflow.log_metric("DGraph AUROC_class" + str(class_no), ROC_test[class_no])
      
    f = open(folder_name + '/Dgraph_model.p','wb') 
    pickle.dump(Dgraph_model.state_dict(),f)
    f.close()
    
    logits_train = Dgraph_model.forward(datainputs_train)
    logits_val = Dgraph_model.forward(datainputs[1])[val_mask]
   
    logits = [logits_train,logits_val, logits_test]
    
    f = open(filepath + '/Dgraph_logits.p','wb') 
    pickle.dump(logits,f)
    f.close()


if __name__ == "__main__":
    
    main(None)
    
    