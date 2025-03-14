#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:15:23 2022

@author: niharika.dsouza
"""
import sys
import json

# Load the configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Add the CODE_PATH to sys.path
sys.path.append(f'{config["CODE_PATH"]}/Graph_Classification/')

import networkx as nx
import scipy.sparse as spp
from sklearn.metrics import roc_curve, auc     
from sklearn.preprocessing import label_binarize
import copy
import seaborn as sns

import collections
import numpy as np
from utils.autoencoder_modules import AE_Model,MLPerceptron
#from joblib import Parallel, delayed #parallel processing

# plotting tools
import matplotlib.pyplot as plt
plt.close('all')


#import deep learning tools
import torch
import dgl

def impute_data(data,mask):
    
    """
    
        Given a dictionary of data features and feature mask, mean impute from the train set
        
        data: dictionary [train,val,test]
        mask: mask [train,val,test]
        
    """
    
    out_data = copy.deepcopy(data)
    
    data_train = data[0]
    mask_train = mask[0]
    
    mean_set = torch.zeros(1,mask_train.shape[1])
    
    for j in range(data_train.shape[1]):
        
        mean_set[0,j] = torch.mean(data_train[mask_train[:,j]==True,j])
        
    for key,ds in enumerate(data):
        
        unimputed = ds
        mask_ds = mask[key]
        
        expanded_mean_set = mean_set.expand_as(unimputed)
        
        for j in range(out_data[key].shape[1]): out_data[key][mask_ds[:,j]==False,j] = expanded_mean_set[mask_ds[:,j]==False,j]
        
    return out_data

def normalise_data(data_ext):
    
    """
    normalise data featurewise according to summary statistics from data_ref
    
    Inputs:
        data_ext: list [train,val,test] or array
    """
    
    #normalises according to training set statistics
    if len((data_ext[0]).size())==2:
        
        data_ref = data_ext[0]
    
        for col in range(data_ref.shape[1]):
        
            if torch.max(data_ref[:,col]) != 1.0 and torch.max(data_ref[:,col]) != 0.0:
            
                minv = torch.min(data_ref[:,col])           
                maxv = torch.max(data_ref[:,col])
                      
                for i in range(len(data_ext)): data_ext[i][:,col] = (data_ext[i][:,col]-minv) / (maxv-minv+1E-5) 
    else:
        
        for col in range(data_ext.shape[1]):
        
            if torch.max(data_ext[:,col]) != 1.0 and torch.max(data_ext[:,col]) != 0.0: #preventthese columns from blowing up
            
                minv = torch.min(data_ext[:,col])           
                maxv = torch.max(data_ext[:,col])
            
                data_ext[:,col] = (data_ext[:,col]-minv) / (maxv-minv+1E-5) 
       
        return data_ext

def prepare_dataset(raw_dataset,data_splits):
    
    """
        Given a dictionary of the data features, prepares data and AE models for training
        
        Inputs:
            
            raw_dataset: dict with feature names as key, each modality is a key type (5 in total: CT, SNP, regimen, clinical, demographic)
            data_splits: list of indices: [train,val,test]
        
        Outputs:
            
            dataset: dict with feature names as key, values is a list of data for [train,val,test]
            models: dict of AE models for each modality with feature name as key 
            modality_dict: list of modalities
            params: training parameters for modality
            
    """
    
    dataset = collections.defaultdict(list)
    mask_set = collections.defaultdict(list)
    models = collections.defaultdict(list)
    params = collections.defaultdict(list)
    modality_dict = []
    
    train_indices = data_splits['train_indices']
    test_indices = data_splits['test_indices']
    val_indices = data_splits['val_indices']
    
    for key,val in enumerate(raw_dataset):
        
        if val == 'demographic_feat':
            
            data_mod = torch.from_numpy((raw_dataset[val][0])).float()
            mask = raw_dataset[val][1]
            
            ds = [data_mod[train_indices],data_mod[val_indices],data_mod[test_indices]] 
            mask_set[val] = [mask[train_indices],mask[val_indices],mask[test_indices]] 
            
            dataset[val] = impute_data(ds,mask_set[val])
            # normalise_data(dataset[val]) #skip this step if unnormalised data is to be used
            
            h_width = 8 # hidden layer dimensions -> modality specific
            model_mod = AE_Model(in_feats= data_mod.size()[1], h_feats=h_width)
            models[val] = model_mod
            
            params[val] = {'lr': 0.0001,'num_epochs' : 50,'batch_size':64}
            #0.001,500,32 -> single layered 500
        
        elif val == 'ct_feat':
            
            data_mod = torch.from_numpy((raw_dataset[val][0])).float()
            mask = raw_dataset[val][1]
            
            ds = [data_mod[train_indices],data_mod[val_indices],data_mod[test_indices]] 
            mask_set[val] = [mask[train_indices],mask[val_indices],mask[test_indices]] 
            
            dataset[val] = impute_data(ds,mask_set[val])
            normalise_data(dataset[val]) #skip this step if unnormalised data is to be used
            
            h_width = 128 # hidden layer dimensions -> modality specific
            model_mod = AE_Model(in_feats= data_mod.size()[1], h_feats=h_width)
            models[val] = model_mod
            
            params[val] = {'lr': 0.0001,'num_epochs' : 150,'batch_size':64}
            #0.0005,300,32 -> single layered (32,500) 150
            
        elif val == 'snp_feat':
            
            data_mod = torch.from_numpy((raw_dataset[val][0])).float()
            mask = raw_dataset[val][1]
            
            ds = [data_mod[train_indices],data_mod[val_indices],data_mod[test_indices]] 
            mask_set[val] = [mask[train_indices],mask[val_indices],mask[test_indices]] 
            
            dataset[val] = impute_data(ds,mask_set[val])
            # normalise_data(dataset[val]) #skip this step if unnormalised data is to be used
            
            h_width = 64 # hidden layer dimensions -> modality specific
            model_mod = AE_Model(in_feats= data_mod.size()[1], h_feats=h_width)
            models[val] = model_mod 
            
            params[val] = {'lr': 0.0001,'num_epochs' : 150,'batch_size':256}
            #0.001,300,32 -> MSE (32m,300), 64
            #0.001,300,32 -> single layered

        
        elif val == 'clinical_feat':
            
            data_mod = torch.from_numpy((raw_dataset[val][0])).float()
            mask = raw_dataset[val][1]
            
            ds = [data_mod[train_indices],data_mod[val_indices],data_mod[test_indices]] 
            mask_set[val] = [mask[train_indices],mask[val_indices],mask[test_indices]] 
            
            dataset[val] = impute_data(ds,mask_set[val])
            # normalise_data(dataset[val]) #skip this step if unnormalised data is to be used
            
            h_width = 128 # hidden layer dimensions -> modality specific rank 178
            model_mod = AE_Model(in_feats= data_mod.size()[1], h_feats=h_width)
            models[val] = model_mod
            
            params[val] = {'lr': 0.0001,'num_epochs' : 250,'batch_size':256}
            #0.00001,300,128 -> single layered

        elif val == 'regimen_feat':
            
            data_mod = torch.from_numpy((raw_dataset[val][0])).float()
            mask = raw_dataset[val][1]
            
            ds = [data_mod[train_indices],data_mod[val_indices],data_mod[test_indices]] 
            mask_set[val] = [mask[train_indices],mask[val_indices],mask[test_indices]] 
            
            dataset[val] = impute_data(ds,mask_set[val])
           # normalise_data(dataset[val]) #skip this step if unnormalised data is to be used
            
            h_width = 64 # hidden layer dimensions -> modality specific  rank 114
            model_mod = AE_Model(in_feats= data_mod.size()[1], h_feats=h_width)
            models[val] = model_mod
            
            params[val] = {'lr': 0.0001,'num_epochs' : 250,'batch_size':128}
            #0.0001,300,128 -> single layered'
            
        if val == 'cont_feat': # features resulting from keeping aside the continuous valued variables
            
            data_mod = torch.from_numpy((raw_dataset[val][0])).float()
            mask = raw_dataset[val][1]
            
            ds = [data_mod[train_indices],data_mod[val_indices],data_mod[test_indices]] 
            mask_set[val] = [mask[train_indices],mask[val_indices],mask[test_indices]] 
            
            dataset[val] = impute_data(ds,mask_set[val])
            normalise_data(dataset[val]) #skip this step if unnormalised data is to be used
            
            h_width = 4 # hidden layer dimensions -> modality specific
            model_mod = AE_Model(in_feats= data_mod.size()[1], h_feats=h_width)
            models[val] = model_mod
            
            params[val] = {'lr': 0.0001,'num_epochs' : 300,'batch_size':64}

            
        if val !=  'outcomes': modality_dict.append(val) #store keys
            
    return dataset,mask_set,models,modality_dict,params

def prepare_modality_nofusion(raw_dataset,data_splits,width):
    
    """
        Given a dictionary of the data features, prepares data and ANN models for training
        
        Inputs:
            
            raw_dataset: dict[0] with feature names as key, each modality is a key type (5 in total: CT, SNP, regimen, clinical, demographic)
                         dict[1] with missing feature locations
            data_splits: list of indices: [train,val,test]
            width: width of MLP hidden layer
        
        Outputs:
            
            dataset: dict with feature names as key, values is a list of data for [train,val,test]
            models: dict of AE models for each modality with feature name as key 
            modality_dict: list of modalities
            params: training parameters for modality
            
    """
    
    dataset = collections.defaultdict(list)
    mask_set = collections.defaultdict(list)
    models = collections.defaultdict(list)
    params = collections.defaultdict(list)
    modality_dict = []
    
    train_indices = data_splits['train_indices']
    test_indices = data_splits['test_indices']
    val_indices = data_splits['val_indices']
    
    for key,val in enumerate(raw_dataset):
        
        if val == 'demographic_feat':
            
            data_mod = torch.from_numpy((raw_dataset[val][0])).float()
            mask = raw_dataset[val][1]
            
            ds = [data_mod[train_indices],data_mod[val_indices],data_mod[test_indices]] 
            mask_set[val] = [mask[train_indices],mask[val_indices],mask[test_indices]] 
            
            dataset[val] = impute_data(ds,mask_set[val])
            # normalise_data(dataset[val]) #skip this step if unnormalised data is to be used
            
            model_mod = MLPerceptron(dataset[val][0].shape[1],width, 5)
            models[val] = model_mod
            
            params[val] = {'lr': 0.0001,'num_epochs' : 150}
        
        elif val == 'ct_feat':
            
            data_mod = torch.from_numpy((raw_dataset[val][0])).float()
            mask = raw_dataset[val][1]
            
            ds = [data_mod[train_indices],data_mod[val_indices],data_mod[test_indices]] 
            mask_set[val] = [mask[train_indices],mask[val_indices],mask[test_indices]] 
            
            dataset[val] = impute_data(ds,mask_set[val])
            normalise_data(dataset[val]) #skip this step if unnormalised data is to be used
            
            model_mod = MLPerceptron(dataset[val][0].shape[1],width, 5)
            models[val] = model_mod
            
            params[val] = {'lr': 0.0001,'num_epochs' : 150}
            
        elif val == 'snp_feat':
            
            data_mod = torch.from_numpy((raw_dataset[val][0])).float()
            mask = raw_dataset[val][1]
            
            ds = [data_mod[train_indices],data_mod[val_indices],data_mod[test_indices]] 
            mask_set[val] = [mask[train_indices],mask[val_indices],mask[test_indices]] 
            
            dataset[val] = impute_data(ds,mask_set[val])
            
            model_mod = MLPerceptron(dataset[val][0].shape[1],width, 5)
            models[val] = model_mod
            
            params[val] = {'lr': 0.0001,'num_epochs' : 150}
        
        elif val == 'clinical_feat':
            
            data_mod = torch.from_numpy((raw_dataset[val][0])).float()
            mask = raw_dataset[val][1]
            
            ds = [data_mod[train_indices],data_mod[val_indices],data_mod[test_indices]] 
            mask_set[val] = [mask[train_indices],mask[val_indices],mask[test_indices]] 
            
            dataset[val] = impute_data(ds,mask_set[val])

            model_mod = MLPerceptron(dataset[val][0].shape[1],width, 5)
            models[val] = model_mod
            
            params[val] = {'lr': 0.0001,'num_epochs' : 150}

        elif val == 'regimen_feat':
            
            data_mod = torch.from_numpy((raw_dataset[val][0])).float()
            mask = raw_dataset[val][1]
            
            ds = [data_mod[train_indices],data_mod[val_indices],data_mod[test_indices]] 
            mask_set[val] = [mask[train_indices],mask[val_indices],mask[test_indices]] 
            
            dataset[val] = impute_data(ds,mask_set[val])

            model_mod = MLPerceptron(dataset[val][0].shape[1],width, 5)
            models[val] = model_mod
            
            params[val] = {'lr': 0.0001,'num_epochs' : 150}
            
        if val == 'cont_feat': # features resulting from keeping aside the continuous valued variables
            
            data_mod = torch.from_numpy((raw_dataset[val][0])).float()
            mask = raw_dataset[val][1]
            
            ds = [data_mod[train_indices],data_mod[val_indices],data_mod[test_indices]] 
            mask_set[val] = [mask[train_indices],mask[val_indices],mask[test_indices]] 
            
            dataset[val] = impute_data(ds,mask_set[val])
            normalise_data(dataset[val]) #skip this step if unnormalised data is to be used
            
            model_mod = MLPerceptron(dataset[val][0].shape[1],width, 5)
            models[val] = model_mod
            
            params[val] = {'lr': 0.0001,'num_epochs' : 150}

            
        if val !=  'outcomes': modality_dict.append(val) #store keys
        
        
    return dataset,models,modality_dict,params