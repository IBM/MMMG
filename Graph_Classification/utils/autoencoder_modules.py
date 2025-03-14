#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:22:32 2022

@author: niharika.dsouza
"""

# import general libraries
# plotting tools

import matplotlib.pyplot as plt
plt.close('all')

#import deep learning tools
import torch
import torch.nn as nn
import torch.nn.functional as F

class AE_Model(nn.Module):
    
    """
    Generic fully connected AutoEncoder
    
    Inputs:
        in_feats: input features
        h_feats: size of encoding
        
    Methods:
        forward: forward pass
    """
    
    def __init__(self, in_feats, h_feats):
        super(AE_Model, self).__init__()
        
        self.enc_dense1 = nn.Linear(in_feats,int(in_feats*0.8))
        self.enc_dense2 = nn.Linear(int(in_feats*0.8),h_feats)
        self.dec_dense1 = nn.Linear(h_feats,int(in_feats*0.8))
        self.dec_dense1.weight = nn.Parameter(self.enc_dense2.weight.transpose(0,1))
        self.dec_dense2 = nn.Linear(int(in_feats*0.8),in_feats)
        self.dec_dense2.weight = nn.Parameter(self.enc_dense1.weight.transpose(0,1))

    
    def forward(self, x):
        
        h = self.enc_dense1(x)
        h = F.leaky_relu(h)
        h = self.enc_dense2(h)
        h = F.leaky_relu(h)
        out = self.dec_dense1(h)
        out = F.leaky_relu(out)
        out = self.dec_dense2(out)
      
        
        return out,h
    
class MLPerceptron(torch.nn.Module):
    
    """
    Generic fully connected MLP
    
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

def normalise(data_ext):
    
    """
    data featurewise according to summary statistics from data_ref
    
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
    
def train_common_encoder(data,h_width):
    
    """
        Train common Encoder
       
        Inputs: 
            complete_dataset: concatenated modality specific embeddings for dataset
           
        Outputs:
            embed_dim: patient embedding from autoencoders
            model: trained model
           
    """
    
    model = AE_Model(in_feats = data[0].size()[1],h_feats=h_width)
    
    lr = 0.0001
    num_epochs = 500
    batch_size = 128
                        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=0.0005)
    
    #prepare/initialise iterables for training
    loss_train = []
    loss_test = []
    loss_val = []
    
    #variables to store embedding
    H_train = []
    H_val = []
    H_test= []
    
    data_train,data_val,data_test = data[0],data[1],data[2]
    train_loader = torch.utils.data.DataLoader(data_train,batch_size =batch_size,shuffle = True,num_workers =0)
        
    criterion = nn.MSELoss(reduction='mean')
    best_val_loss = criterion(model(data_val)[0],data_val)
    best_test_loss = criterion(model(data_test)[0],data_test)
    
    print('\n Train Common Encoder')
 
    for epoch in range(num_epochs):
            
        running_loss = 0.0
            
        for i,data_train_batch in enumerate(train_loader,0):
                
            inputs, targets = data_train_batch,data_train_batch
                
            model.train()
            outputs = model(inputs)[0]
            
            loss = criterion(targets,outputs)
                
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
              
            running_loss +=loss.item()
             
            
        loss_train.append(running_loss)
                          #/i)
            
        model.eval() 
            
        with torch.no_grad():
         
                loss_val.append(criterion(data_val,model(data_val)[0]))
                loss_test.append(criterion(data_test,model(data_test)[0]))
                
        if epoch>0:
                
            #check for best performance 
                
            best_val_loss = min(best_val_loss,loss_val[epoch])
            if (best_val_loss == loss_val[epoch]): best_test_loss = loss_test[epoch] 
            
        #print training statistics every 5 epochs
        if epoch % 5 == 0:
             print('\n In epoch {}, loss: {:.5f}, val loss: {:.5f} (best {:.5f}), test loss: {:.5f} (best {:.5f})'.format(
                        epoch, loss_train[epoch], loss_val[epoch], best_val_loss, loss_test[epoch], best_test_loss))  
        
    H_train = model(data_train)[1]
    H_val = model(data_val)[1]
    H_test = model(data_test)[1]
    
    embed_dim = [H_train,H_val,H_test]
    

    return [embed_dim,model]


def train_modality_encoder(complete_dataset, modality_models, modality_dict, params):
    
    """
       Train Modality Specific Encoders
       
       Inputs: 
           complete_dataset: entire dataset
           modality_models: individual AEs for each modality
           modality_dict: list of modalities
           
       Outputs:
           embed_dim: list of embeddings from autoencoders: [train,val,test]
           modality_models: trained models
           
    """
    
   #variables to store embeddings
    H_train = [] 
    H_test =  []
    H_val = []
    
    for key,val in enumerate(modality_dict):
        
        model = modality_models[val] #select models
        
        #select data and split into train,val,test      
        data = complete_dataset[val]
        data_train,data_val,data_test = data[0],data[1],data[2]

        #training specifics
        
        lr = params[val]['lr']        
        num_epochs = params[val]['num_epochs']
        batch_size = params[val]['batch_size']
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=0.005)
        
        print('\n Training AE for Modality : ' + str(key) + '|| ' + val)
        
        #prepare/initialise iterables for training
        loss_train = []
        loss_test = []
        loss_val = []
        
        train_loader = torch.utils.data.DataLoader(data_train,batch_size =batch_size,shuffle = True,num_workers =0)
       
        criterion = torch.nn.MSELoss(reduction='mean')
        best_val_loss = criterion(model(data_val)[0],data_val)
        best_test_loss = criterion(model(data_test)[0],data_test)
        
        
        for epoch in range(num_epochs):
            
            running_loss = 0.0
            
            for i,data_train_batch in enumerate(train_loader,0):
                
                inputs, targets = data_train_batch,data_train_batch
                
                model.train()
                outputs = model(inputs)[0]
                
                loss = criterion(targets,outputs)
                
                #backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss +=loss.item()
            
            loss_train.append(running_loss)
                              #/i)
            
            model.eval() 
            
            with torch.no_grad():

                loss_val.append(criterion(data_val,model(data_val)[0]))
                loss_test.append(criterion(data_test,model(data_test)[0]))
                
    
                
            if epoch>0:
                
                #check for best performance 
                
                best_val_loss = min(best_val_loss,loss_val[epoch])
                if (best_val_loss == loss_val[epoch]): best_test_loss = loss_test[epoch] 
            
            #print training statistics every 5 epochs
            if epoch % 5 == 0:
                print('\n In epoch {}, loss: {:.5f}, val loss: {:.5f} (best {:.5f}), test loss: {:.5f} (best {:.5f})'.format(
                        epoch, loss_train[epoch], loss_val[epoch], best_val_loss, loss_test[epoch], best_test_loss))
        
        #store embeddings, normalise before concatenation
                        
        normalised_data = normalise(model(data_train)[1])
        H_train.append(normalised_data.detach())
        
        normalised_data = normalise(model(data_val)[1])
        H_val.append(normalised_data.detach())
        
        normalised_data = normalise(model(data_test)[1])
        H_test.append(normalised_data.detach())
 
    #create embeddings
    embed_dim_train = H_train[0]
    embed_dim_val = H_val[0]
    embed_dim_test = H_test[0]
    
    for i in range(1,len(H_train)): embed_dim_train = torch.cat((embed_dim_train,H_train[i]), dim=1)
    
    embed_dim_val = H_val[0]
    for i in range(1,len(H_val)): embed_dim_val = torch.cat((embed_dim_val,H_val[i]), dim=1)
    
    embed_dim_test = H_test[0]
    for i in range(1,len(H_test)): embed_dim_test = torch.cat((embed_dim_test,H_test[i]), dim=1)
    
    embed_dim= [embed_dim_train,embed_dim_val,embed_dim_test]
    
    return embed_dim,modality_models

        