#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 00:18:03 2022

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
from utils.baseline_fusion_utils import concat_features, MLPerceptron, train_classifier
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
    
    mlflow.set_experiment("Graph Classification: Early Fusion and Intermediate Fusion")
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
    
    filename = folder_name + 'Front_end_representation.p'

    with open(filename, 'rb') as f:
           x = pickle.load(f)
    f.close()

    embeddings_concat = x['embeddings_concat']

    filename = folder_name + '/Common_encoder_trained.p'

    with open(filename, 'rb') as f:
           x = pickle.load(f)
    f.close()

    embeddings_common = x['embeddings_common']
    common_model = x['models']
    outcomes = x['outcomes']
    data_splits = x['data_splits']
    
    print("Early Fusion")
    [dataset,mask_set,models,modality_dict,params] = prepare_dataset(TB_dataset, data_splits)
    features = concat_features(dataset,data_splits) # early fusion
   
    ANN_model = MLPerceptron(features[0].shape[1], 500, num_labels)
    
    [ANN_model,logits] = train_classifier(ANN_model,features,outcomes, lr =0.0001,num_epochs = 150)
    logits_test = ANN_model(features[2])
    roc_EF = performance_evaluate(outcomes[2], logits_test, 'ANN Early Fusion ', folder_name)
    
    for class_no in range(len(roc_EF)):
        mlflow.log_metric("EF_AUROC_class " + str(class_no), roc_EF[class_no])
        print("EF_AUROC_class " + str(class_no) + ': ' + str(roc_EF[class_no]))
        
    print("Intermediate Fusion \n")
    ANN_common_model = MLPerceptron(embeddings_concat[0].shape[1], 100, num_labels)
       
    [ANN_common_model,logits_common] = train_classifier(ANN_common_model,embeddings_concat,outcomes, lr= 0.001, num_epochs = 100)
    
    #performance evaluate
    logits_test = F.softmax(ANN_common_model(embeddings_concat[2]),dim=1)
    roc_IF = performance_evaluate(outcomes[2], logits_test, 'ANN Intermediate Fusion ', folder_name)
    
    for class_no in range(len(roc_IF)):
        mlflow.log_metric("IF_AUROC_class " + str(class_no), roc_IF[class_no])
        print("IF_AUROC_class " + str(class_no) + ': ' + str(roc_IF[class_no]))


if __name__ == "__main__":
    
    main(None)
    
    
   
    
    