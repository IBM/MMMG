#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 22:39:54 2025

@author: niharika.dsouza
"""

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# plotting tools
import matplotlib.pyplot as plt
plt.ioff()
plt.close('all')

class Metric_Fusion_Network(nn.Module):
      
    """
    Metric_Fusion_Network : Performs Late Fusion according to https://www.sciencedirect.com/science/article/pii/S1361841523003249#b4
    
    Inputs:
    
        h_feats: hidden layer width
        modality_dict: dictionary of modality names
        num_classes: number of outcomes to be predicted
        lambda_1 : loss parameter
        
    Use simple linear projection 
    """
    
    def __init__(self, in_feats, h_feats, modality_dict, num_classes):
        
        super(Metric_Fusion_Network, self).__init__()
        
       
        self.num_classes = num_classes
        self.h_feats = h_feats

        self.lin_proj1 =  [nn.Linear(modality_dict[v],self.h_feats,bias=True) for k,v in enumerate((modality_dict))]  
        self.lin_proj2 = [nn.Linear(self.h_feats, self.h_feats,bias=True) for k,v in enumerate((modality_dict))]
        self.lin_proj3 = [nn.Linear(self.h_feats, self.num_classes,bias=True) for k,v in enumerate((modality_dict))]
        
        self.bn1 =  [nn.BatchNorm1d(self.h_feats) for k,v in enumerate((modality_dict))]

         
        self.linears1 = nn.ModuleList(self.lin_proj1)
        self.linears2 = nn.ModuleList(self.lin_proj2)
        self.linears3 = nn.ModuleList(self.lin_proj3)
        
        self.M = len(modality_dict)
        self.drop = nn.Dropout(0.2)
        
    def forward(self, feats):
        
        """
        Inputs:
            feats: dictionary of modality features
         Outputs:
                out: projected features
         """
    
        proj_feat = collections.defaultdict(list)
    
        for k,v in enumerate((feats)):
            features = torch.zeros(feats[v].shape[0],self.num_classes)
            continue
            
        for k,v in enumerate((feats)):
            #all features to common space using individual projections
                h = self.lin_proj1[k](feats[v])
                h = F.leaky_relu(h) 
                h = self.lin_proj2[k](h)
                h = F.leaky_relu(h) 
                h = self.lin_proj3[k](h)

                proj_feat[v] = h
                features += h
                        
        out = features/self.M
        
        
        return proj_feat,out
    
    def trainer(self,data,targets,val_mask,lambda_1,batch_size,lr,num_epochs):
    
        """
        trains a Multimodal Fusion framework with the Metric Loss from cheerla
    
        Inputs:
            model: instantiated model of type Metric_Net
            data: dictionary [train,val], each being a dictionary of multimodal data
            targets: dictionary [train,val] of outcomes
            val_mask: [N_train+N_val,1] binary mask for validation dictionary to indicate which subjects are in the validation set 
            lambda_1: tradeoff between HGR and CE term
            batch_size: batch_size during training
            lr: learning rate (initial)
            num_epochs: number of epochs to train
    
        """
    
        data_train,data_val = data[0],data[1]
        targets_train,targets_val = torch.from_numpy(targets[0]).long(),torch.from_numpy(targets[1]).long()

        #training specifics
        flag = (lambda_1<1.0)
    
        for k,v in enumerate(data_train):
            N = data_train[v].shape[0]
            continue
    
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.001)
     
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
                    eff_bs = len(indices)
                
                else:
                
                    indices = ptation[j*batch_size:(j+1)*batch_size].astype(int)
                    eff_bs = batch_size
                    
                self.train()
            
                datainputs_batch = collections.defaultdict(list)
            
                for k,v in enumerate(data_train):
                    datainputs_batch[v] = data_train[v][indices,:]
            
                targets_train_batch = targets_train[indices]
            
                outputs_f, logits_train_batch = self.forward(datainputs_batch)
         
                #compute metric loss
                loss = self.loss(outputs_f, eff_bs, targets_train_batch,lambda_1)
            
                #backward pass
                optimizer.zero_grad()
            
                loss.backward()
                optimizer.step()
       
                running_loss +=loss.item()
                
                del datainputs_batch
        
            print("Epoch %d done, running validation stats ... running loss %1.3f" %(epoch,running_loss/num_loop))
            self.eval() 
        
            with torch.no_grad():
            
                outputs_val,logits_val = self.forward(data_val)
                datainputs_val = collections.defaultdict(list)
                datainputs_train = collections.defaultdict(list)
                train_mask = val_mask ==False
            
                for k,v in enumerate(outputs_val):
                
                    datainputs_val[v] = outputs_val[v][val_mask,:]
                    datainputs_train[v] = outputs_val[v][train_mask,:]
            
            
                logits_train = logits_val[train_mask,:]
                logits_val = logits_val[val_mask,:]
            
                n_tr = targets_train.shape[0]
                n_val = targets_val.shape[0]    
        
            
                loss_train_epoch = self.loss(datainputs_train, n_tr, targets_train, lambda_1).item()
                loss_val_epoch = self.loss(datainputs_val, n_val, targets_val, lambda_1).item()
            
                loss_val.append(loss_val_epoch)
                loss_train.append(loss_train_epoch)

            
                if epoch ==0 :
                    best_val_loss = loss_val[0]
                
                pred_train = logits_train.argmax(1)
                pred_val = logits_val.argmax(1)
                
                train_acc = (pred_train == targets_train).float().mean()
                val_acc = (pred_val == targets_val).float().mean()
               
        
        
            if epoch>0:
            
                #check for best performance  
                best_val_loss = min(best_val_loss,loss_val[epoch])
        
            print(' In epoch {}, loss: {:.5f} ,acc: {:.3f}, val loss: {:.5f} (best {:.5f}), val acc {:.3f} '.format(
                     epoch, loss_train[epoch], train_acc, loss_val[epoch], best_val_loss, val_acc))

    
#         plt.figure(figsize=(10,5))
#         plt.plot(loss_train,'r',label= 'train combined') 
#         plt.plot(loss_val,'g',label = 'val combined')
#         plt.xlabel('Epochs')
#         plt.ylabel('Total Loss')
#         plt.legend(loc='upper right')
#         plt.show()
#         plt.close()
    
    
    def loss(self,outputs_batch,batch_size,Y,lambda_1):
    
        """
        Computes metric loss for batch according to https://www.sciencedirect.com/science/article/pii/S1361841523003249#b4
        
        Inputs: 
            outpus_batch- dictionary of batched outputs
            batch_size - size of batch
            Y, batch targets
            lambda_1: parameter from metric loss
        
        Outputs:
            Metric Loss
        """
        loss_sim = 0
        loss_cel = 0
        
        M = len(outputs_batch)
        N = batch_size
        
        G_mat = torch.zeros([N,N])
        cos = torch.nn.CosineSimilarity( dim=-1, eps=1e-6)
        
        for k,v in enumerate(outputs_batch):
           
            f_m = outputs_batch[v]
    
            loss_cel += F.cross_entropy(f_m, Y)
            G_mat += cos(f_m[None,:,:],f_m[:,None,:]) #self modality pairs
         
            for k1,v1 in enumerate(outputs_batch):
            
                # avoid double counting during the loop
                if not(k1 < k):
              
                    g_m = outputs_batch[v1]
                    G_mat += cos(f_m[None,:,:],g_m[:,None,:])
                
            
        loss_sim += ( -torch.sum(torch.sum(G_mat)) + 2* torch.trace(G_mat))/(M**2 * N**2)
        loss_sim = torch.max(torch.tensor([0.0]), lambda_1 + loss_sim)
                   
        return loss_sim + loss_cel/(M)