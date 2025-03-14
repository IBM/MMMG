#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:19:04 2022

@author: niharika.dsouza
"""

import sys,os
import json

# Load the configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Add the CODE_PATH to sys.path
sys.path.append(f'{config["CODE_PATH"]}/Graph_Classification/')

from utils.data_prep import prepare_dataset
from utils.autoencoder_modules import train_common_encoder,train_modality_encoder
from utils.graph_creation_utils import convert_to_multigraphs

import numpy as np
import torch,os,sys
import pickle,random
import collections

# plotting tools
import matplotlib.pyplot as plt
plt.close('all')

def main(run_number):
    
    random.seed(run_number)
    
    #load from saved pickle file
    
    with open(f'{config["CODE_PATH"]}/TB_data.pickle','rb') as f:
         TB_dataset = pickle.load(f)
    f.close()
    
    N = 3051 #number of patients for TB dataset
    thresh_q = 0.9 #threshold for graph generation
    h_width = 32 # concept encoder width
    
    folder_name = f'{config["SAVE_PATH"]}/Patient_Graphs/H_' + str(h_width) + '/' 
    filepath  = folder_name + '/' + str(thresh_q) + '/' + str(run_number) + '/'
    
    if not (os.path.exists(folder_name)):
        os.makedirs(folder_name)
    
    data_splits = collections.defaultdict(list)    
    permutations = np.random.permutation(N)
    
    data_splits['train_indices'] = permutations[:int(0.7*N)]
    data_splits['val_indices'] = permutations[int(0.7*N):int(0.8*N)]
    data_splits['test_indices'] = permutations[int(0.8*N):]

    #data preparation
    [dataset,mask,models,modality_dict,params] = prepare_dataset(TB_dataset, data_splits)
    
    #train modality specific encoders
    [embeddings_concat,models] = train_modality_encoder(dataset, models, modality_dict, params)
    
    #save models
    dict_modality_specific = {'embeddings_concat': embeddings_concat, 'models':models}
    
    if not (os.path.exists(filepath)):
        os.makedirs(filepath)
        
    f = open(filepath + 'Front_end_representation.p','wb')
    pickle.dump(dict_modality_specific,f)
    f.close()
    
    
    #train common encoder
    [embeddings_common,common_model] = train_common_encoder(embeddings_concat,h_width)
      
    #outcomes
    data_mod = TB_dataset['outcomes']
    train_indices = data_splits['train_indices']
    test_indices = data_splits['test_indices']
    val_indices = data_splits['val_indices']
    outcomes = [data_mod[train_indices],data_mod[val_indices],data_mod[test_indices]]
    
    if not (os.path.exists(filepath+'/Train/')):
        
        os.makedirs(filepath+'/Train/')
        os.makedirs(filepath+'/Test/')
        os.makedirs(filepath+'/Val/')

    dict_common = {'embeddings_common': embeddings_common, 'models':common_model, 'outcomes': outcomes,
                              'data_splits':data_splits}
    f = open(filepath+'/Common_encoder_trained.p','wb')
    
    pickle.dump(dict_common,f)
    f.close()
    
    N_types = embeddings_concat[0].shape[1]
    E_types = h_width
    
    #creates_multigraphs
    convert_to_multigraphs(embeddings_concat,embeddings_common,common_model,thresh_q,folder_name,run_number)
    

if __name__ == "__main__":
    
    main()
    
