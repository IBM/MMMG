#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:24:27 2025

@author: niharika.dsouza
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl,collections
from dgl.nn import GINConv
import dgl.function as fn
import numpy as np
import matplotlib.pyplot as plt

class HGR_Network(nn.Module):
    """
          HGR_Network : performing Hirschfield-Gabor-Renyi metric based alignment of modality features
    
            Inputs:
                h_dim: dimensionality of HGR decomposition
                h_feats: hidden layer width for MultiGraph Neural Network/MLP
                modality_dict: dictionary of modality input sizes
                num_classes: number of outcomes to be predicted
                drop_rate: quantile for thresholding HGR correlation (retains a sparse multi-layered graph connectivity)
        
            Use simple MLP for projection
    
            Methods: 
      
            _________________
                Max_Corr_sparsify, create_multigraph:
        
               Implements differentiable block wise graph sparsification and graph sturcture learning from https://link.springer.com/chapter/10.1007/978-3-031-47679-2_11
           
               To be used in conjunction with gnn_forward

            ------------------
            gnn_predictor,gnn_forward:
          
               Implements a multi-GNN predictor from https://link.springer.com/chapter/10.1007/978-3-031-47679-2_11
           
               Inputs:
                  features: feature dictionary with keys as modality names
      
            ------------------     
            mlp_predictor,mlp_forward:
        
               Implements an MLP predictor from baseline https://ojs.aaai.org/index.php/AAAI/article/view/4464
           
               Inputs:
                  features: feature dictionary with keys as modality names
      
            -------------------
         
            soft_HGR_loss, trainer
        
    """
    def __init__(self, in_feats, h_feats, modality_dict, num_classes, drop_rate):
        
        super(HGR_Network, self).__init__()
            
        self.in_feats = in_feats
        self.num_classes = num_classes
            
            #modality wise MLPs for modality feature projection
        self.lin_proj1 =  [nn.Linear(modality_dict[v],32,bias=True) for k,v in enumerate((modality_dict))]  
        self.lin_proj2 = [nn.Linear(32, 32,bias=True) for k,v in enumerate((modality_dict))]
        self.lin_proj3 = [nn.Linear(32, self.in_feats,bias=True) for k,v in enumerate((modality_dict))]
        self.linears1 = nn.ModuleList(self.lin_proj1)
        self.linears2 = nn.ModuleList(self.lin_proj2)
        self.linears3 = nn.ModuleList(self.lin_proj3)
            
        self.bn1 =  [nn.BatchNorm1d(self.in_feats) for k,v in enumerate((modality_dict))]
        self.M = len(modality_dict)
        self.drop_rate = drop_rate
        self.h_feats = h_feats
            
    def HGR_forward(self,feats):
            
            proj_feat = collections.defaultdict(list)
            features = torch.zeros(1,self.in_feats)

            for k,v in enumerate((feats)):
                #projects all features to common space using individual projections
                h = self.lin_proj1[k](feats[v])
                h = F.leaky_relu(h) 
                h = self.lin_proj2[k](h)
                h = F.leaky_relu(h) 
                h = self.lin_proj3[k](h)

                proj_feat[v] = h
                features = torch.cat([features,h],dim=0)

            return proj_feat,features[1:,:]

    def Max_Corr_sparsify(self):

            self.c = torch.nn.Parameter(torch.Tensor(1,self.M)) #across node copies
            torch.nn.init.xavier_uniform(self.c)

            self.sparse = (torch.nn.Parameter(torch.Tensor(self.M*(self.M-1),1))) 
            torch.nn.init.xavier_uniform(self.sparse)
            self.sparse_m = F.sigmoid(self.sparse)

    def gnn_predictor(self):

            self.linAC1 = nn.Linear(self.in_feats,self.h_feats)
            self.convAC1 = GINConv(self.linAC1, 'mean')    

            self.linCA1 = nn.Linear(self.in_feats,self.h_feats)
            self.convCA1 = GINConv(self.linCA1,'mean') 

            self.linAC2 = nn.Linear(2*self.h_feats, self.h_feats)
            self.convAC2 = GINConv(self.linAC2,'mean')       

            self.linCA2 = nn.Linear(2*self.h_feats, self.h_feats)
            self.convCA2 = GINConv(self.linCA2,'mean')  

            #batch norm for GNN
            self.bn_gnn1 =  nn.BatchNorm1d(2*self.h_feats)
            self.bn_gnn2 =  nn.BatchNorm1d(2*self.h_feats)

            #for classifier
            self.lin_readout  = nn.Linear(2*self.h_feats, self.num_classes,bias=False)

    def create_multigraph(self,bs,nnodes,A_W, num_mod,dr):

            """
            Parameters
            ----------
            bs : current mini batch size.
            nnodes : effective number of nodes.
            A_W : HGR correlation matrix.
            num_mod : number of modalities
            dr : drop rate for thresholding

            Returns
            -------
            g1: graph based on Supra-walk matrix AC
            g2: graph based on Supra-walk matrix CA

            """

            A = torch.zeros((bs,bs))
            C = torch.eye(bs)

            #extract edge weights for within and across modalities from correlation matrix
            #use stats from each block to maintain desired level of sparsity

            count = 0
           
            for i in range(num_mod):
                for j in range(num_mod):
          
                    if j>=i:

                        count+=1
                        a_W_block = A_W[i*nnodes:(i+1)*nnodes , j*nnodes:(j+1)*nnodes]
    
                        mask_block = dr[count]
                        a_W_block = F.relu(a_W_block-mask_block)

                    if (i==j): #self blocks

                        A[i*nnodes:(i+1)*nnodes , j*nnodes:(j+1)*nnodes] = a_W_block

                    else:

                        C[i*nnodes:(i+1)*nnodes , j*nnodes:(j+1)*nnodes] = a_W_block
                        C[j*nnodes:(j+1)*nnodes , i*nnodes:(i+1)*nnodes] = a_W_block.transpose(0,1)


            AC = A.mm(C)
            # adding num_nodes to avoid running into the situation where disconnected nodes are 
            # dropped completely 
            # normalise adjacency matrices for stable forward pass
            #normalise the adjacency matrices and create graphs
            edge_r,edge_c = torch.nonzero(AC,as_tuple = True)

            g1 = dgl.graph((edge_r,edge_c),num_nodes = bs)
            degs = g1.in_degrees().float()

            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0

            g1.ndata['norm'] = norm.unsqueeze(1)
            g1.apply_edges(fn.u_mul_v('norm', 'norm', 'normalized'))


            g2 = dgl.graph((edge_c,edge_r), num_nodes = bs)
            degs = g2.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            g2.ndata['norm'] = norm.unsqueeze(1)
            g2.apply_edges(fn.u_mul_v('norm', 'norm', 'normalized'))

            print(str(len(edge_r)) + ' edges retained')


            return [g1,g2]

    def mlp_predictor(self):

            self.fc1 = nn.Linear( self.in_feats *self.M, self.h_feats)
            self.fc2 = nn.Linear( self.h_feats, self.num_classes)

    def gnn_forward(self, features):

            bs = features.shape[0] #current batch size
            nnodes = int(bs/self.M) #number of nodes in batch

            #estimate full correlation matrix for mini-batch, then create sparse graph
            A_W = torch.abs(torch.corrcoef(features)) - torch.eye((bs))
            [g1,g2] = self.create_multigraph(bs,nnodes,A_W,self.M,self.sparse_m)     


            #Add edge weighted importance during forward pass          
            h_AC1 = F.relu(self.convAC1(g1,(features),edge_weight= g1.edata['normalized']))
            h_AC1 = h_AC1.reshape(g1.number_of_nodes(),-1)
            h_CA1 = F.relu(self.convCA1(g2,(features),edge_weight= g2.edata['normalized']))
            h_CA1 = h_CA1.reshape(g2.number_of_nodes(),-1)

            h = (torch.cat((h_AC1,h_CA1),dim=1))
            h = self.bn_gnn1(h)

            h_AC2 = F.relu(self.convAC2(g1,h,edge_weight=g1.edata['normalized']))
            h_AC2 = h_AC2.reshape(g1.number_of_nodes(),-1)
            h_CA2 = F.relu(self.convCA2(g2,h,edge_weight=g2.edata['normalized']))
            h_CA2 = h_CA2.reshape(g2.number_of_nodes(),-1)

            # aggregate across transition types 
            h = (torch.cat((h_AC2,h_CA2),dim=1))
            h = self.bn_gnn2(h)

            # project classes
            h = self.lin_readout(h)
            #aggregate across modalities using a weighted sum, this reduces the number of learnable parameters
            agg_nc = torch.kron(F.softmax(self.c,dim=1),torch.eye(nnodes))       
            h = agg_nc.mm(h)

            return h

    def mlp_forward(self, proj_feats):
            
            keys = list(proj_feats.keys())
            features = torch.zeros(proj_feats[keys[0]].shape[0],self.in_feats)
            # concatenate all features from common space 

            for k,v in enumerate(proj_feats):
                
                features = torch.cat([features,proj_feats[v]],dim=1)
            
            #slice correctly and pass through MLP predictor
            out = self.fc1(features[:,self.in_feats:])
            out = F.relu(out)
            out = self.fc2(out)
            out = F.relu(out)

            return out

    def soft_HGR_loss(self,outputs_batch,batch_size):

            """
            Inputs: 
                outptus_batch- dictionary of batched outputs

            Outputs:
                soft HGR loss for batch
            """
            covs_trace = 0 # trace term for covariance
            exp_term = 0 # expectation term

            M = len(outputs_batch)

            for k,v in enumerate(outputs_batch):

                f_m = torch.mean(outputs_batch[v], dim = 0).reshape(1,-1)           
                N = outputs_batch[v].shape[1]


                f_ms = outputs_batch[v] - f_m.expand_as(outputs_batch[v])
                f_cov = torch.cov(f_ms.transpose(0,1)) + 0.001* torch.eye(N) # add reg term for stability

                for k1,v1 in enumerate(outputs_batch):

                    # avoid double counting during the loop
                    if not(k1 <= k):

                        g_m = torch.mean(outputs_batch[v1], dim = 0).reshape(1,-1)
                        g_ms = outputs_batch[v1] - g_m.expand_as(outputs_batch[v1])
                        g_cov = torch.cov(g_ms.transpose(0,1)) + 0.001* torch.eye(N)

                        covs_trace += 0.5 * torch.trace(f_cov.mm(g_cov))

                        # using diag and sum instead of trace helps make the computation much faster
                        exp_norm_term  = torch.diag((f_ms).mm(g_ms.transpose(0,1)))   
                        exp_term += torch.sum(exp_norm_term)*(1/(batch_size-1))
            # return negative of HGR loss for minimisation
            # normalisation is determined by how many such terms are there - M(M-1) and the number of summations in the trace, i.e. M - use normalised correlation instead?

            return (-exp_term  + covs_trace)/(M*(M-1))

    def trainer(self, data, targets, val_mask,lambda_1, batch_size, lr, num_epochs, pred_flag):

        """
        trains a Multimodal Fusion framework with the soft HGR loss across modalities and multi-layered GNN/ANN for outcome prediction, allows for staged training of projection and prediction networks

        Inputs:
            self: instantiated model of type HGR_Network
            data: dictionary [train,val], each being a dictionary of multimodal data
            targets: dictionary [train,val] of outcomes
            val_mask: [N_train+N_val,1] binary mask for validation dictionary to indicate which subjects are in the validation set 
            lambda_1: tradeoff between HGR and CE loss terms
            batch_size: batch_size during training
            lr: learning rate (initial)
            num_epochs: number of epochs to train

            pred_flag : 'GNN', 'MLP' to indicate mode of training 

        """

        data_train,data_val = data[0],data[1]
        targets_train,targets_val = torch.from_numpy(targets[0]).long(),torch.from_numpy(targets[1]).long()

        #training specifics

        for k,v in enumerate(data_train):
            N = data_train[v].shape[0]
            continue

                #instantiate
        if pred_flag == 'GNN':

            self.Max_Corr_sparsify()
            self.gnn_predictor()
            self.lambda_1 = lambda_1

        elif pred_flag == 'MLP':

            self.mlp_predictor()
            self.lambda_1 = lambda_1
        
        else:
            
            print("Predictor type incorrectly specified, please check usage and try again")

        # Freeze projection layers
        if lambda_1 == 0.0:
            for m in self.linears1:
                  m.requires_grad = False
            for m in self.linears2:
                  m.requires_grad = False
            for m in self.linears3:
                  m.requires_grad = False


        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.001)

        #prepare/initialise iterables for training
        loss_train = []
        loss_val = []

        loss_train_hgr = []
        loss_val_hgr = []

        loss_train_ce = []
        loss_val_ce = []

        num_loop = int(np.ceil(N/batch_size))
        ptation = np.random.permutation(np.arange(N))

        #training loop
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
                [outputs_proj,feats] = self.HGR_forward(datainputs_batch)

                if pred_flag == 'GNN': 
                    logits_train_batch = self.gnn_forward(feats)
                else:
                    logits_train_batch = self.mlp_forward(outputs_proj)

                cel = F.cross_entropy(logits_train_batch,targets_train_batch)
                
                if lambda_1 == 0.0:

                    loss = cel 

                else:   
                    sgl = self.soft_HGR_loss(outputs_proj,eff_bs) #soft HGR loss 
                    loss = lambda_1*sgl + (1-lambda_1)*cel  #combined

                #backward pass
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                running_loss +=loss.item()

                del datainputs_batch

            print("Epoch %d done, running validation stats ... running loss %1.3f" %(epoch,running_loss/num_loop))
            self.eval() 

            with torch.no_grad():

                [outputs_val,feats_val] = self.HGR_forward(data_val)

                if pred_flag == 'GNN': 
                    logits_val = self.gnn_forward(feats_val)
                else:
                    logits_val = self.mlp_forward(outputs_val)

                datainputs_val = collections.defaultdict(list)
                datainputs_train = collections.defaultdict(list)

                train_mask = val_mask ==False

                for k,v in enumerate(outputs_val):

                    datainputs_val[v] = outputs_val[v][val_mask,:]
                    datainputs_train[v] = outputs_val[v][train_mask,:]


                logits_train = logits_val[train_mask,:]
                logits_val = logits_val[val_mask,:]

                n_tr = targets_train.shape[0]

                sgl_t = self.soft_HGR_loss(datainputs_train,n_tr)
                cel_t = F.cross_entropy(logits_train,targets_train)

                n_val = targets_val.shape[0]

                sgl_v = self.soft_HGR_loss(datainputs_val,n_val)
                cel_v= F.cross_entropy(logits_val,targets_val)

                print("Loss breakdown|| train HGR %1.3f , train CE %1.3f , val HGR %1.3f , val CE %1.3f "%(sgl_t,cel_t,sgl_v,cel_v))


                loss_train_epoch = lambda_1*sgl_t + (1-lambda_1) * cel_t
                loss_val_epoch = lambda_1*sgl_v +(1-lambda_1) * cel_v

                loss_val.append(loss_val_epoch)
                loss_train.append(loss_train_epoch)

                loss_val_hgr.append(sgl_v)
                loss_val_ce.append(cel_v)

                loss_train_hgr.append(sgl_t)
                loss_train_ce.append(cel_t)

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

            # early stopping
            if cel_v < 1.00 and epoch >50:
                break

#         plt.figure(figsize=(10,5))
#         plt.plot(loss_train,'r',label= 'train combined') 
#         plt.plot(loss_val,'g',label = 'val combined')
#         plt.xlabel('Epochs')
#         plt.ylabel('Total Loss')
#         plt.legend(loc='upper right')
#         plt.show()
#         plt.close()


#         plt.figure(figsize=(10,5))
#         plt.plot(loss_train_hgr, 'r', label = 'train HGR')
#         plt.plot(loss_val_hgr, 'g', label = 'val HGR')
#         plt.xlabel('Epochs')
#         plt.ylabel('HGR Loss')
#         plt.legend(loc='upper right')
#         plt.show()
#         plt.close()


#         plt.figure(figsize=(10,5))
#         plt.plot(loss_train_ce, 'r', label = 'train CE') 
#         plt.plot(loss_val_ce, 'g', label = 'val CE')
#         plt.xlabel('Epochs')
#         plt.ylabel('Cross Entropy Loss')
#         plt.legend(loc='upper right')
#         plt.show()
#         plt.close()


