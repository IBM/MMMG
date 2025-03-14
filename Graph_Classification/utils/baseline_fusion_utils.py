#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 00:13:47 2022

@author: niharika.dsouza
"""

from sklearn import svm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from utils.data_prep import performance_evaluate

class MLPerceptron(torch.nn.Module):
    
    """
    Generic fully connected Neural Network
    
    Inputs:
        in_feats: input features
        h_feats: size of encoding
        num_classes: no of outcomes
        
    Methods:
        forward: forward pass
    """
    
    def __init__(self,in_feats, h_feats, num_classes):
        super(MLPerceptron, self).__init__()
        
        self.fc1 = nn.Linear(in_feats,h_feats)
        self.relu = torch.nn.LeakyReLU() # instead of Heaviside step fn    def forward(self, x):
        self.fc2 = nn.Linear(h_feats,20)
        self.fc3 = nn.Linear(20,num_classes)
        
    def forward(self,x):
        
        output = self.relu(self.fc1(x)) # instead of Heaviside step fn
        output = self.relu(self.fc2(output))
        output = self.relu(self.fc3(output))
        
        return output
    
def concat_features(dataset,data_splits):
    
    """
    Prepare for early fusion
    
    Parameters
    ----------
    dataset : [train,val,test] dictionary of modality specific features
    Returns
    -------
    features: [train,val,test] concatenated
    """
    
    train_indices = data_splits['train_indices']
    test_indices = data_splits['test_indices']
    val_indices = data_splits['val_indices']
    
    train_feat = torch.zeros(train_indices.size,1)
    test_feat = torch.zeros(test_indices.size,1)
    val_feat = torch.zeros(val_indices.size,1)
    
    for key,val in enumerate(dataset):
    
            train_feat  = torch.cat((train_feat,dataset[val][0]),dim=1)
            val_feat  = torch.cat((val_feat,dataset[val][1]),dim=1)
            test_feat  = torch.cat((test_feat,dataset[val][2]),dim=1)
        
    features = [train_feat[:,1:],val_feat[:,1:],test_feat[:,1:]]

    return features
    
def train_classifier(model,concat_inputs, targets, lr, num_epochs):
    """
     Train MLP classifier fully supervised setting
     
     Inputs:
         model: initialised ANN model
         concat_inputs: input features
         targetrs: outcomes
         lr:learning rate
         num_epochs: number of epochs
         
    Outputs:
         model: trained ANN model
         logits: [train,val,test] logits from training
                  
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.999),eps=1e-08)
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