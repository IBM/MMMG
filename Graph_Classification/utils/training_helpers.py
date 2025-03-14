#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 22:53:34 2022

@author: niharika.dsouza
"""
import sys
import json

# Load the configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Add the CODE_PATH to sys.path
sys.path.append(f'{config["CODE_PATH"]}/Graph_Classification/')

import torch,collections
import numpy as np
import torch.nn.functional as F
# plotting tools
import matplotlib.pyplot as plt
plt.close('all')
from utils.evaluators import performance_evaluate

def train_multiplex(node_features,graphs,targets,model,lr,num_epochs,M,num_labels):

    """
    training the multiplex model
    
    Inputs:
        graphs: inputs [train,val,test]
        node_features: inputs [train,val,test]
        model: Neural Network 
        targets: labels [train,val,test]
        lr: learning rate
        num_epochs: number of training epochs
        M: number of levels in graph
    Output:
        model: trained model
    """   
  
    
    # #initialize model and optimizer
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,factor=0.99,min_lr =0.0001,cooldown = 10)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma=0.1)
    
    #store losses for training
    best_val_acc = 0
    
    loss_train = torch.zeros([num_epochs,1]) 
    loss_val = torch.zeros([num_epochs,1]) 
    
    model.train()
    
    logits_train = torch.zeros(len(graphs[0]), num_labels)
    targets_train = torch.tensor(targets[0],dtype=torch.int64)
    
    batch_size = 128
    num_loop = int(np.ceil(len(graphs[0])/batch_size))
    
    for epoch in range(num_epochs):
        
        ptation = np.random.permutation(len(graphs[0]))
        
        running_loss = 0
        
        for j in range(num_loop):
            
            print("Epoch %s || Mini batch %s / %s" %(epoch, j, num_loop))
            
            if j== num_loop-1:
                
                indices = ptation[j*batch_size:].astype(int)
                graphs_batch = [graphs[0][index] for index in indices]
                
                logits_train_batch = logits_calculate_multiplex(graphs_batch, node_features[0][indices,],model,num_labels,M)
                targets_train_batch = torch.tensor(targets[0][indices],dtype=torch.int64)
                
                logits_train[j*batch_size:,] = logits_train_batch.data 
                targets_train[j*batch_size:] = targets_train_batch.data
                
                
            else:
            
                indices = (ptation[j*batch_size:(j+1)*batch_size]).astype(int)
                graphs_batch = [graphs[0][index] for index in indices]
                
                logits_train_batch =  logits_calculate_multiplex(graphs_batch, node_features[0][indices,],model,num_labels,M)
                targets_train_batch = torch.tensor(targets[0][indices],dtype=torch.int64)
                
                logits_train[j*batch_size:(j+1)*batch_size,] = logits_train_batch.data 
                targets_train[j*batch_size:(j+1)*batch_size] = targets_train_batch.data
                
        #logits_train = logits_calculate_multiplex(graphs[0][800:],node_features[0][800:,:],model,num_labels,M)  
            loss = F.cross_entropy(logits_train_batch,targets_train_batch)
            
            model.train()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            running_loss+=loss.data/num_loop
            del logits_train_batch
        
        loss_train[epoch]=running_loss
        
        #clear variables for memory        
        loss = 0
        running_loss = 0
        
        model.eval()
        
        # model evaluation
        with torch.no_grad():
            
            print("Estimating validation stats for epoch %s" % (epoch))
            
            logits_val = logits_calculate_multiplex(graphs[1],node_features[1],model,num_labels,M).data
            targets_val = torch.tensor(targets[1],dtype=torch.int64)
            val_loss = F.cross_entropy(logits_val,targets_val)
            loss_val[epoch] = val_loss.data
        
        
        #compute accuracies
        pred_train = logits_train.argmax(1)
        pred_val = logits_val.argmax(1)

        train_acc = (pred_train == targets_train).float().mean()
        val_acc = (pred_val == targets_val).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            #best_test_acc = test_acc 
            
        if epoch % 1 == 0:
            print('Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f)' % (
                loss_train[epoch],
                train_acc.item(),
                val_acc.item(),
                best_val_acc
                ))
        
        scheduler.step()

    #plot loss curve
#     plt.figure(figsize=(10,5))
#     plt.plot(loss_train.detach().numpy(),'r',label= 'train') 
#     plt.plot(loss_val.detach().numpy(),'g',label = 'val')
      
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Combined')
#     plt.legend()
        
#     plt.show()
#     plt.close() 
    
    return model

def train_mG(node_features,graphs,targets,model,lr,num_epochs,M,num_labels):

    """
    Trains on multiplex-like graphs - use for mGNN and MultiBehavioral GNN
    
    Inputs:
        graphs: list of input graphs [train,val,test]
        node_features: inputs [train,val,test]
        model: Neural Network
        targets: labels [train,val,test]
        lr: learning rate
        num_epochs: number of training epochs
        M: number of levels in graph
    Output:
        model: trained model
    """   
    
    # #initialize model and optimizer
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma=0.5)
    
    #store losses for training
    best_val_acc = 0
    best_test_acc = 0
    
    loss_train = torch.zeros([num_epochs,1]) 
    loss_test = torch.zeros([num_epochs,1])   
    loss_val = torch.zeros([num_epochs,1]) 
    
    model.train()
    
    logits_train = torch.zeros(len(graphs[0]), num_labels)
    targets_train = torch.tensor(targets[0],dtype=torch.int64)
    
    batch_size = 32
    num_loop = int(np.ceil(len(graphs[0])/batch_size))
    
    for epoch in range(num_epochs):
        
        ptation = np.random.permutation(len(graphs[0]))
        
        running_loss = 0
        
        for j in range(num_loop):
            
            if j== num_loop-1:
                
                indices = ptation[j*batch_size:].astype(int)
                graphs_batch = [graphs[0][index] for index in indices]
                
                logits_train_batch = logits_calculate_mG(graphs_batch, node_features[0][indices,],model,num_labels,M)
                targets_train_batch = torch.tensor(targets[0][indices],dtype=torch.int64)
                
                logits_train[j*batch_size:,] = logits_train_batch.data 
                targets_train[j*batch_size:] = targets_train_batch.data
                
                
            else:
            
                indices = (ptation[j*batch_size:(j+1)*batch_size]).astype(int)
                graphs_batch = [graphs[0][index] for index in indices]
                
                logits_train_batch =  logits_calculate_mG(graphs_batch, node_features[0][indices,],model,num_labels,M)
                targets_train_batch = torch.tensor(targets[0][indices],dtype=torch.int64)
                
                logits_train[j*batch_size:(j+1)*batch_size,] = logits_train_batch.data 
                targets_train[j*batch_size:(j+1)*batch_size] = targets_train_batch.data
                
            loss = F.cross_entropy(logits_train_batch,targets_train_batch)
            
            model.train()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss+=loss.data/num_loop
        
        #optimize
        loss_train[epoch]=running_loss.data
        
        #clear variables for memory        
        loss = 0
        running_loss = 0
        
        model.eval()
        
        # model evaluation
        with torch.no_grad():
            logits_val = logits_calculate_mG(graphs[1],node_features[1],model,num_labels,M)   
            targets_val = torch.tensor(targets[1],dtype=torch.int64)
            val_loss = F.cross_entropy(logits_val,targets_val)
            loss_val[epoch] = val_loss.data 
            
            logits_test = logits_calculate_mG(graphs[2],node_features[2],model,num_labels,M) 
            targets_test = torch.tensor(targets[2],dtype=torch.int64)
            loss_test[epoch] = F.cross_entropy(logits_test,targets_test).data 
        
        #compute accuracies
        pred_train = logits_train.argmax(1)
        pred_val = logits_val.argmax(1)
        pred_test = logits_test.argmax(1)

        train_acc = (pred_train == targets_train).float().mean()
        val_acc = (pred_val == targets_val).float().mean()
        test_acc = (pred_test == targets_test).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc 
            
        if epoch % 1 == 0:
            print('Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
                loss_train[epoch],
                train_acc.item(),
                val_acc.item(),
                best_val_acc.item(),
                test_acc.item(),
                best_test_acc.item(),
                ))
        

        scheduler.step()

    #plot loss curve
#     plt.figure(figsize=(10,5))
#     plt.plot(loss_train.detach().numpy(),'r',label= 'train') 
#     plt.plot(loss_val.detach().numpy(),'g',label = 'val')

#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Combined')
#     plt.legend()
        
#     plt.show()
#     plt.close() 
    
    return model

def train_gcn(node_features,graphs,targets,model,lr,num_epochs,num_labels):

    """
    Trains GCNs on input data
    
    Inputs:
        graphs: inputs [train,val,test]
        node_features: inputs [train,val,test]
        model: Neural Network Predictor
        targets: labels [train,val,test]
        lr: learning rate
        num_epochs: number of training epochs
        M: number of levels in graph
    Output:
        model: trained model
    """  

    # #initialize model and optimizer
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=5e-04)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma=0.5)

    #store losses for training
    best_val_acc = 0
    best_test_acc = 0
    
    loss_train = torch.zeros([num_epochs,1]) 
    loss_test = torch.zeros([num_epochs,1])   
    loss_val = torch.zeros([num_epochs,1]) 
    
    logits_train = torch.zeros(len(graphs[0]), num_labels)
    targets_train = torch.tensor(targets[0],dtype=torch.int64)
    
    batch_size = 32
    num_loop = int(np.ceil(len(graphs[0])/batch_size))
    
    for epoch in range(num_epochs):
        
        ptation = np.random.permutation(len(graphs[0]))
        model.train()
        running_loss = 0
        
        for j in range(num_loop):
            
            print("Epoch %s || Mini batch %s / %s" %(epoch, j, num_loop))
             
            if j== num_loop-1:
                
                indices = ptation[j*batch_size:].astype(int)
                graphs_batch = [graphs[0][index] for index in indices]
                
                logits_train_batch = logits_calculate(graphs_batch, node_features[0][indices,],model,num_labels)
                targets_train_batch = torch.tensor(targets[0][indices],dtype=torch.int64)
                
                logits_train[j*batch_size:,] = logits_train_batch.data 
                targets_train[j*batch_size:] = targets_train_batch.data
                           
            else:
            
                indices = (ptation[j*batch_size:(j+1)*batch_size]).astype(int)
                graphs_batch = [graphs[0][index] for index in indices]
                
                logits_train_batch =  logits_calculate(graphs_batch, node_features[0][indices,],model,num_labels)
                targets_train_batch = torch.tensor(targets[0][indices],dtype=torch.int64)
                
                logits_train[j*batch_size:(j+1)*batch_size,] = logits_train_batch.data 
                targets_train[j*batch_size:(j+1)*batch_size] = targets_train_batch.data
                
            loss = F.cross_entropy(logits_train_batch,targets_train_batch)
            
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            running_loss+=loss.data/num_loop
        
        #optimize
        
        loss_train[epoch]=running_loss        
        # model evaluation
        
        model.eval()
        
        with torch.no_grad():
            logits_val = logits_calculate(graphs[1],node_features[1],model,num_labels)    
            targets_val = torch.tensor(targets[1],dtype=torch.int64)
            loss_val[epoch] = F.cross_entropy(logits_val,targets_val)
            
            logits_test = logits_calculate(graphs[2],node_features[2],model,num_labels)    
            targets_test = torch.tensor(targets[2],dtype=torch.int64)
            loss_test[epoch] = F.cross_entropy(logits_test,targets_test)
        
        #compute accuracies
        pred_train = logits_train.argmax(1)
        pred_val = logits_val.argmax(1)
        pred_test = logits_test.argmax(1)

        train_acc = (pred_train == targets_train).float().mean()
        val_acc = (pred_val == targets_val).float().mean()
        test_acc = (pred_test == targets_test).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc 
            
        if epoch % 1 == 0:
            print('Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
                loss_train[epoch],
                train_acc.item(),
                val_acc.item(),
                best_val_acc.item(),
                test_acc.item(),
                best_test_acc.item(),
                ))
        
        scheduler.step()
        
    #plot loss curve
#     plt.figure(figsize=(10,5))
#     plt.plot(loss_train.detach().numpy(),'r',label= 'train') 
#     plt.plot(loss_val.detach().numpy(),'g',label = 'val')
      
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Combined')
#     plt.legend()
        
#     plt.show()
#     plt.close() 
    
    return model

def train_latentgraph(model,data,targets,val_mask,batch_size,lr,num_epochs):
    
    """
    trainer for latent graph learning
    
    Inputs:
        model: instantiated model of latent graph learning type
        data: dictionary [train,val], each being a dictionary of multimodal data
        targets: dictionary [train,val] of outcomes
        val_mask: [N_train+N_val,1] binary mask for validation dictionary to indicate which subjects are in the validation set or not, info from these to be excluded during training 
        batch_size: batch_size during training
        lr: learning rate (initial)
        num_epochs: number of epochs to train
    Outputs:
        model: trained model
    """
    
    data_train,data_val = data[0],data[1]
    targets_train,targets_val = torch.from_numpy(targets[0]).long(),torch.from_numpy(targets[1]).long()
    
    N = data_train.shape[0]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)
    
    #prepare/initialise iterables for training
    loss_train = []
    loss_val = [] 
    num_loop = int(np.ceil(N/batch_size))
    ptation = np.random.permutation(np.arange(N))
    
    for epoch in range(num_epochs):
        
        running_loss = 0.0
        
        for j in range(num_loop):
            
            print("Running batch %d of %d" %(j+1,num_loop))
            
            if j== num_loop-1:
                indices = ptation[j*batch_size:].astype(int)              
            else:         
                indices = ptation[j*batch_size:(j+1)*batch_size].astype(int)

                     
            #train model
            model.train()
            
            datainputs_batch = collections.defaultdict(list)
            datainputs_batch = data_train[indices,:]
            
            targets_train_batch = targets_train[indices]
            logits_train_batch = model(datainputs_batch)
            
            loss = F.cross_entropy(logits_train_batch,targets_train_batch)
            
            #backward pass
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
       
            running_loss +=loss.item()
            
            del datainputs_batch #save memory
        
        print("Epoch %d done, running validation stats ... running loss %1.3f" %(epoch,running_loss/num_loop))
        model.eval() 
        
        with torch.no_grad():
            
            logits_val = model(data_val)
            train_mask = val_mask ==False
            
            logits_train = logits_val[train_mask,:]
            logits_val = logits_val[val_mask,:]
                     
            cel_t = F.cross_entropy(logits_train,targets_train)
            cel_v = F.cross_entropy(logits_val,targets_val)
                    
            loss_train_epoch =  cel_t
            loss_val_epoch =  cel_v
           
            loss_val.append(loss_val_epoch)
            loss_train.append(loss_train_epoch)
          
            if epoch ==0 :
                best_val_loss = loss_val[0]
                
            pred_train = logits_train.argmax(1)
            pred_val = logits_val.argmax(1)
            train_acc = (pred_train == targets_train).float().mean()
            val_acc = (pred_val == targets_val).float().mean()
               
        scheduler.step()
        
        if epoch>0:
            
            #check for best performance  
            best_val_loss = min(best_val_loss,loss_val[epoch])
        
        print(' In epoch {}, loss: {:.5f} ,acc: {:.3f}, val loss: {:.5f} (best {:.5f}), val acc {:.3f} '.format(
                     epoch, loss_train[epoch], train_acc, loss_val[epoch], best_val_loss, val_acc))
        
        # early stopping criteria to prevent overfitting problem
        if cel_v < 0.9 and epoch >50:
            break
    
#     plt.figure(figsize=(10,5))
#     plt.plot(loss_train,'r',label= 'train combined') 
#     plt.plot(loss_val,'g',label = 'val combined')
#     plt.xlabel('Epochs')
#     plt.ylabel('Total Loss')
#     plt.legend(loc='upper right')
#     plt.show()
#     plt.close()

    return model

def train_modality_predictors(dataset,models,params,outcomes,folder_name):
    
    """
    given the modality dataset and MLP models, train the modality specific predictors for outcome classification
    
    Inputs:
        dataset: dictionary of inputs, the key is one of 6 modalities [train,val,test]
        models: dictionary of models
        params: dictionary of parameters for each modality
        outcomes: ground truth labels [train,val,test]
        folder_name: folder for storing AUC plots
   
    Outputs:
        logits_all: logits for splits [train,val,test] each item of list is a dictionary indexed by val
        models: dictionary of models
    """
    
    
    logits_test = collections.defaultdict(list)
    logits_train = collections.defaultdict(list)
    logits_val = collections.defaultdict(list)
    roc = collections.defaultdict(list)

    for key, val in enumerate(dataset):
        
        print('Training Modality '+ val)
        input_data = dataset[val]
        model_MLP = models[val]
        params_MLP = params[val]
        
        [model_MLP,l] = train_vanilla_classifier(model_MLP,input_data,outcomes, lr=params_MLP['lr'], num_epochs = params_MLP['num_epochs'])
            
        l_test = F.softmax(model_MLP(input_data[2]),dim=1)
        l_train = F.softmax(model_MLP(input_data[0]),dim=1)
        l_val = F.softmax(model_MLP(input_data[1]),dim=1)
        roc[val] =  performance_evaluate(outcomes[2], l_test, 'ANN on modality : ' + val,folder_name)
        
        logits_test[val] = l_test
        logits_train[val] = l_train
        logits_val[val] = l_val
        
        models[val] = model_MLP
    
    logits_all = [logits_train,logits_val,logits_test]
    
    return logits_all,models,roc

def train_vanilla_classifier(model,concat_inputs, targets, lr, num_epochs):
    """
     Train MLP classifier fully supervised setting
     
     Inputs:
         model: initialised MLP model
         concat_inputs: input features
         targetrs: outcomes
         lr:learning rate
         num_epochs: number of epochs
         
    Outputs:
         model: trained MLP model
         logits: [train,val,test] logits from training
                  
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,betas=(0.9,0.999),eps=1e-08,weight_decay=0.5)
    best_val_acc = 0
    best_test_acc = 0

    train_feats = concat_inputs[0]
    val_feats = concat_inputs[1]
    test_feats = concat_inputs[2]
    
    targets_train = torch.tensor(targets[0],dtype=torch.int64)
    targets_val = torch.tensor(targets[1],dtype=torch.int64)
    targets_test = torch.tensor(targets[2],dtype=torch.int64)

    
    for e in range(num_epochs):
        
        # Forward pass
        logits_train = model(train_feats)

        # Compute prediction
        pred_train = logits_train.argmax(1)
        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits_train,targets_train)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy on training/validation/test
        with torch.no_grad():
            
            logits_val = model(val_feats)
            logits_test = model(test_feats)
            
            pred_val = logits_val.argmax(1)
            pred_test = logits_test.argmax(1)

            train_acc = (pred_train== targets_train).float().mean()
            val_acc = (pred_val== targets_val).float().mean()
            test_acc = (pred_test== targets_test).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc     

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, train acc: {:.3f} , val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc))    
         
    logits = [logits_train,logits_val,logits_test]    
    
    return [model,logits]

def logits_calculate(graphs,node_features,model,num_classes):
    
    """
    Inputs:
        model: RGCN/MultiDim GCN model
        graphs: multigraphs
        node_features: input features
        num_classes: no of outcomes
    
    Outputs:
        logits:predicted logits
    
    """
    logits = torch.zeros(len(graphs),num_classes)
    
    for pat_no in range(node_features.shape[0]):
            
            G = graphs[pat_no]       
            G.features = node_features[pat_no].view(-1,1)
            logits[pat_no] = model(G)
   
    return logits

def logits_calculate_multiplex(graphs,node_features,model,num_classes,M):
    
    """
    calculate logits for multiplex
    
    Inputs:
        model: multiplex model
        graphs: multiplex graph
        node_features: input features
        num_classes: no of outcomes
        M: no of multiplex layers
    
    Outputs:
        logits:predicted logits
    
    """
    logits = torch.zeros(len(graphs),num_classes)
    
    for pat_no in range(node_features.shape[0]):
            
            G = graphs[pat_no]   
            logits[pat_no] = model(G,M,node_features[pat_no].view(-1,1))
   
    return logits

def logits_calculate_mG(graphs,node_features,model,num_classes,M):
    
    """
    Inputs:
        model: multibehav/mGNN model
        graphs: multibehav/mGNN graph
        node_features: input features
        num_classes: no of outcomes
        M: no of multiplex layers
    Outputs:
        logits:predicted logits
    
    """
    logits = torch.zeros(len(graphs),num_classes)
    
    for pat_no in range(node_features.shape[0]):
            
            G = graphs[pat_no]   
            logits[pat_no] = model(G[0],G[1],M,node_features[pat_no].view(-1,1))             
   
    return logits