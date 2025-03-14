#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:02:44 2022

@author: niharika.dsouza
"""

import matplotlib.pyplot as plt
plt.close('all')

#import deep learning tools
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GINConv,GATConv,GraphConv,RelGraphConv


class MultiplexGNN(nn.Module):
      
    """
    Multiplex GNN framework for message passing via multiplex walks
    
    Inputs:
         N: number of nodes in multiplex
        in_feats: input features 
        h_feats: hidden layer width
        num_classes: number of outcomes
        M: number of multiplex layers /edge types
        
    GINConv layers can be replaced with other conv layers such as sGCN Conv, SAGE Conv etc
    
    Outputs:
        Graph label logits
    """
    
    def __init__(self, g1, in_feats, num_classes, M):
        
        super(MultiplexGNN, self).__init__()
        
        self.in_feats = in_feats
        self.num_classes = num_classes
        
        self.convAC1 = GINConv(F.leaky_relu, 'mean')        
        self.convCA1 = GINConv(F.leaky_relu, 'mean')
             
        self.convAC2 = GINConv(F.leaky_relu, 'mean')        
        self.convCA2 = GINConv(F.leaky_relu, 'mean') 
        
        self.agg = torch.nn.Linear(4*in_feats,1)
        self.dp_out = torch.nn.Dropout(0.5)
        
        self.dense_1 = torch.nn.Linear(g1.number_of_nodes(),100)
        self.dense_2 = torch.nn.Linear(100,20)
        self.dense_3 = torch.nn.Linear(20,num_classes)
        
    def forward(self, g1, M, node_features):
        
        
        #type II graph can be inferred from type I graph
        g2 = dgl.from_scipy(g1.adj(scipy_fmt='coo').T)
         
        #replicate features across node copies
        features = torch.kron(torch.ones(M,1),node_features)
           
        h_AC1 = self.convAC1(g1,(features))
        h_CA1 = self.convCA1(g2,(features))

        h = (torch.cat((h_AC1,h_CA1),dim=1))
        
        h_AC2 = self.convAC2(g1,h)
        h_CA2 = self.convCA2(g2,h)
       
        # #aggregate across features
        h = torch.cat((h_AC2,h_CA2),dim=1)
        
        h = self.agg(h)
        
        h = self.dp_out(h)
        
        # graph based readout
        h  = F.leaky_relu(self.dense_1(h.transpose(0,1)))
        h  = F.leaky_relu(self.dense_2(h))
        h  = self.dense_3(h)

        return h
    
    

class MultiDimGCN_stack(nn.Module):
    
    """
    Multidimensional Graph Convolutional Network in https://arxiv.org/pdf/1808.06099.pdf
    
    Inputs:
        in_size: input message dimensional 
        hidden size: hidden layer dimensionality
        out_size : output dimensionality
        
    Output:
        Graph label logits
        
    """
    def __init__(self, G, in_size, hidden_size, out_size, num_classes,alpha):
        super(MultiDimGCN_stack, self).__init__()
        
        self.hidden_size = hidden_size 
        self.alpha = alpha
        self.out_size = out_size
       
        self.M = len(G.canonical_etypes)
        
        # create message passing layers
        self.layer1 = MultiDimGCN_Layer(in_size, hidden_size, G.etypes, G.ntypes, alpha)
        # self.layer2 = MultiDimGCN_Layer(hidden_size, out_size , G.etypes, G.ntypes, alpha)     
        
        self.dense_1 = torch.nn.Linear(G.features.shape[0]*out_size,100)
        self.dense_2 = torch.nn.Linear(100,20)
        self.dense_3 = torch.nn.Linear(20,num_classes)
        
    def forward(self, G):
        
        embed_dict = {ntype : G.features
                      for ntype in G.ntypes}
             
        
        h_dict = self.layer1(G, embed_dict)
        h_dict = {k :  h for k, h in h_dict.items()}
               
        # h_dict = self.layer2(G, h_dict)
        # h_dict = {k : h for k, h in h_dict.items()}
        
        h_dict  = self.dense_1(h_dict['feat'].reshape(-1,self.out_size*G.features.shape[0]))
        h_dict = F.leaky_relu(self.dense_2(h_dict))
        h_dict = self.dense_3(h_dict)
        
        # get logits for graph
        return h_dict.reshape(1,-1)
    
class MultiDimGCN_Layer(torch.nn.Module):
    
    """
    message passing layer for Multidimensional GCN from https://arxiv.org/pdf/1808.06099.pdf
    
    Inputs:
        in_size: message dimensions
        out_size : target dimensions
        etype: types of edges
        ntype: type of nodes
        
    Outputs: 
        dictionary of node features corresponding to different types of nodes (our case ntypes =1)
    """
    
    def __init__(self,  in_size, out_size, etypes, ntypes, alpha):
        super(MultiDimGCN_Layer, self).__init__()
    
        
        # W_r for each relation  
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, in_size) for name in etypes
            })
       
        # init bilinear attentional weight
        self.M_att = nn.Linear(in_size, in_size)
        
        self.weight_readout =   nn.ModuleDict({
                name : nn.Linear(in_size * len(etypes), out_size) for name in ntypes })
        
        self.etypes = etypes
        self.alpha = alpha
        
        
    def forward(self, G, feat_dict):
        
        funcs = {}
        
        for srctype, etype, dsttype in G.canonical_etypes:

            Wh = self.weight[etype] # weight
            # Compute W_r * h
            Wh_x = F.leaky_relu(Wh(feat_dict[srctype])) #layer-wise features before message passing
            
            #Compute regular graph convolution cross layer graph convolution
            G.nodes[srctype].data['Wh_x_%s' % etype] = Wh_x  * self.alpha
            
            #function for regular forward pass
            funcs[etype] = (fn.copy_u('Wh_x_%s' % etype, 'm'), fn.mean('m', 'h')) 
            
            #computing cross layered attentions
            p_vec = torch.zeros(len(self.etypes),1) #storing attentions
            
            for num_ep ,etype_pair in enumerate(self.etypes):            
                #bilinear attention computation
                Wd = self.weight[etype_pair]
                p_vec[num_ep] = torch.trace(Wd((Wh(self.M_att.weight)).transpose(0,1)))   

            # layer specific attentions
            p_vec = F.softmax(p_vec) # attention vector normalised
            Wh_cross = torch.zeros(Wh_x.size()[0],1)
            
            for num_ep , etype_pair in  enumerate(self.etypes):
                
                Wh_cross = Wh_cross + p_vec[num_ep] * Wh_x   #cross contributions    
                
            G.nodes[srctype].data['Wh_cross_%s' % etype] = Wh_cross * (1- self.alpha) 
         
        #stack features, this is like an R-GCN layer without aggregation
        G.multi_update_all(funcs, 'stack')
        
        #message passing across layers includes cross dependencies
        x = torch.zeros(G.features.shape[0],1)     
        for srctype, etype, dsttype in G.canonical_etypes:       
            y = G.nodes[srctype].data['Wh_cross_%s' % etype].reshape(G.features.shape[0],-1)
            x = torch.cat((x,y),1)
        
        #linear readout at layer output
        for ntype in G.ntypes: 
            x = x[:,1:] + G.nodes[ntype].data['h'].reshape(G.features.shape[0],-1)
            G.nodes[ntype].data['h']  = self.weight_readout[ntype](x) #final layer readout
        
        return {ntype : (G.nodes[ntype].data['h'])
                for ntype in G.ntypes}
    

    
class MultiBehavioralGNN(torch.nn.Module):
    
    
    """
    Multibehavioral GNN framework for message passing from https://dl.acm.org/doi/pdf/10.1145/3340531.3412119
    
    Inputs:
        g1: Quotient Graph
        g2: Multiplex Graph
        in_feats: input features
        h_feats: hidden layer width
        num_classes: number of outcomes
        M: number of multiplex layers /edge types
        
    Outputs:
        Graph label logits
        
    """
    
    def __init__(self, g1, g2, in_feats, h_feats, num_classes,M):
        
        super(MultiBehavioralGNN, self).__init__()
        
        self.in_feats = in_feats
        self.num_classes = num_classes

        
        self.convQ1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)  
        self.convM1 = GraphConv(h_feats, h_feats, allow_zero_in_degree=True)  
        
        self.M = M

        self.agg = torch.nn.Linear(h_feats,1)
        self.dpout = torch.nn.Dropout(p=0.5)

        
        self.dense_1 = torch.nn.Linear(g2.number_of_nodes(),100)
        self.dense_2 = torch.nn.Linear(100,20)
        self.dense_3 = torch.nn.Linear(20,num_classes)
        
    def forward(self, g1, g2, M, node_features):
       
        
        Q_node_features = node_features
                    
        h_Q1 = F.leaky_relu(self.convQ1(g1,(Q_node_features)))
        
        #node based feature stack for Multiplex graph
        M_node_features = torch.kron(torch.ones(M,1),h_Q1)
        h_M1 = F.leaky_relu(self.convM1(g2,(M_node_features)))
    
        #graph level readout
        h = self.agg(h_M1)
        
        #graph based prediction
        h = self.dpout(h)
        h  = F.leaky_relu(self.dense_1(h.transpose(0,1)))
        h  = F.leaky_relu(self.dense_2(h))     
        h  = self.dense_3(h)

        return h


class mGNN(torch.nn.Module):
    
    
    """
    mGNN framework for message passing from https://arxiv.org/abs/2109.10119
    
    Inputs:
        g1: intra-graph
        g2: inter-graph
        in_feats: input features
        h_feats: hidden layer width
        num_classes: number of outcomes
        M: number of multiplex layers /edge types
    
    Outputs:
        
        Graph label logits
        
     """
    
    def __init__(self, g1, g2, in_feats, h_feats, att_heads, num_classes,M):
        
        super(mGNN, self).__init__()
        
        self.in_feats = in_feats
        self.num_classes = num_classes

        
        self.convC1 = GATConv(in_feats, in_feats, num_heads=att_heads, allow_zero_in_degree= True)        
        self.convA1 = GATConv(in_feats, in_feats, num_heads=att_heads, allow_zero_in_degree= True)        
        
             
        self.convC2 = GATConv(2*in_feats*att_heads, h_feats, num_heads=att_heads, allow_zero_in_degree= True)               
        self.convA2 = GATConv(2*in_feats*att_heads, h_feats, num_heads=att_heads, allow_zero_in_degree= True)        
        
        self.dp_out = torch.nn.Dropout(0.5)
        self.agg = torch.nn.Linear(2*h_feats*att_heads,1)
        
        self.dense_1 = torch.nn.Linear(g1.number_of_nodes(),100)
        self.dense_2 = torch.nn.Linear(100,20)
        self.dense_3 = torch.nn.Linear(20,num_classes)
        
    def forward(self, g1, g2, M, node_features):
       

        features = torch.kron(torch.ones(M,1),node_features)
             
        h_C1 = self.convC1(g1,(features)).reshape(features.size()[0],-1)
        h_A1 = self.convA1(g2,(features)).reshape(features.size()[0],-1)

        h = F.leaky_relu(torch.cat((h_C1,h_A1),dim=1))
        
        h_C2 = self.convC2(g1,h).reshape(features.size()[0],-1)
        h_A2 = self.convA2(g2,h).reshape(features.size()[0],-1)

        # graph based readout
        h_1 = (torch.cat((h_C2,h_A2),dim=1))


        #aggregate messages from layers       
        h = self.agg(h_1)
        h = self.dp_out(h)
        
        #graph readout
        h  = F.leaky_relu(self.dense_1(h.transpose(0,1)))
        h  = F.leaky_relu(self.dense_2(h))
        h  = self.dense_3(h)
        #
        return h       

class Relational_GCN(torch.nn.Module):
    
    """
    Relational GCN in https://arxiv.org/pdf/1703.06103.pdf
    (Uses in-built Relational Graph Conv 
    
    Inputs:
        in_size: input features
        hidden size: hidden layer dimensionality
        num_classes: number of classes
        
    Output:
        Graph label logits
        
    """
    def __init__(self, G, in_size, hidden_size, num_classes):
        super(Relational_GCN, self).__init__()
        
        self.hidden_size = hidden_size 
        self.num_bases = 8
        
        # create layers
        self.layer1 = RelGraphConv(in_size, hidden_size, num_rels= len(G.etypes),regularizer = "basis",
                num_bases = self.num_bases)
        self.layer2 = RelGraphConv(hidden_size, hidden_size, num_rels= len(G.etypes),regularizer = "basis",
                 num_bases = self.num_bases)
        
        self.agg = torch.nn.Linear(hidden_size,1)
        self.dense_1 = torch.nn.Linear(G.features.shape[0],100)
        self.dense_2 = torch.nn.Linear(100,20)
        self.dense_3 = torch.nn.Linear(20,num_classes)
        
        self.dp_out = torch.nn.Dropout(0.5)
        
    def forward(self, G):
        
        embed_dict = G.features
        
        etype = dgl.to_homogeneous(G).edata[dgl.ETYPE]
        
        h = self.layer1(dgl.to_homogeneous(G), embed_dict, etype)
        h = F.leaky_relu(h)
        
        h = self.layer2(dgl.to_homogeneous(G), h, etype)
        h = self.agg(h)

        h = self.dp_out(h)
        h  = self.dense_1(h.reshape(-1,G.features.shape[0]))
        h = F.leaky_relu(self.dense_2(h))
        h = self.dense_3(h)

        return h.reshape(1,-1)
    
class GCN_BL(torch.nn.Module):
    
    """
    Baseline Graph Conv
    
    Inputs:
        in_size: input features (node features + M degree features)
        hidden size: hidden layer dimensionality
        num_classes: number of classes
        
    Output:
        Graph label logits
        
    """
    def __init__(self, G, in_size, hidden_size, num_classes):
        super(GCN_BL, self).__init__()
        
        self.hidden_size = hidden_size 
        
        # create layers
        self.layer1 = GraphConv(in_size, hidden_size, allow_zero_in_degree=True)
        self.layer2 = GraphConv(in_size, hidden_size, allow_zero_in_degree=True)
        
        self.agg = torch.nn.Linear(hidden_size,1)
        self.dense_1 = torch.nn.Linear(G.features.shape[0],100)
        self.dense_2 = torch.nn.Linear(100,20)
        self.dense_3 = torch.nn.Linear(20,num_classes)
        
        self.dp_out = torch.nn.Dropout(0.5)
        
    def forward(self, G):
    
        
        embed_dict = G.features
        
        h = self.layer1(G, embed_dict)
        h = F.leaky_relu(h)
        
        h = self.layer2(G, h)
        h = self.agg(h)

        h = self.dp_out(h)
        h  = self.dense_1(h.reshape(-1,G.features.shape[0]))
        h = F.leaky_relu(self.dense_2(h))
        h = self.dense_3(h)

        return h.reshape(1,-1)
    
    
class DGraph_GAT(nn.Module):
      
    """
    DGraph GAT : https://arxiv.org/abs/2003.13620 
    
    Inputs:
        in_feats: input features (node features + 1 optional degree features for walk graph)
        h_feats: hidden layer width
        num_classes: number of outcomes to be predicted
        
    Uses a GAT like model for latent graph learning (implementation not available)
    """
    
    def __init__(self, in_feats, h_feats, num_classes):
        
        super(DGraph_GAT, self).__init__()
        
        self.in_feats = in_feats
        self.num_classes = num_classes
        self.h_feats = h_feats
        
        # trainable parameters controlling linkages
        self.t = torch.nn.Parameter(torch.Tensor(1,1))
        self.theta = torch.nn.Parameter(torch.Tensor(1,1)) 
        
        torch.nn.init.xavier_uniform(self.t)
        torch.nn.init.xavier_uniform(self.theta)
    
        self.mlp1 = nn.Linear(in_features=self.in_feats, out_features=self.h_feats)
        self.mlp2 = nn.Linear(in_features=self.h_feats, out_features=self.h_feats)
        self.mlp3 = nn.Linear(in_features=self.h_feats, out_features=32)
        
        self.gconv1 = GraphConv(32,16,norm='right')
        self.gconv2 = GraphConv(16,8,norm='right')
        
        self.lin1 = nn.Linear(in_features=8, out_features=16)
        self.lin2 = nn.Linear(in_features=16, out_features=self.num_classes)
        self.drop = nn.Dropout(p=0.1)
        
        
    def forward(self, x):
         
        n,w = x.size()
         
        h = self.mlp1(x)
        h = self.drop(h)
        h = F.relu(h)
         
        h = self.mlp2(h)
        h = self.drop(h)
        h = F.relu(h)
         
        h = self.mlp3(h)
         
         
        exp_term = self.t*(torch.cdist(h,h) + self.theta)
        A = F.sigmoid(exp_term)
         
        edge_r,edge_c = torch.nonzero(A,as_tuple = True)
        g = dgl.graph((edge_r,edge_c),num_nodes = n)
         
        h = self.gconv1(g,h,edge_weight=A[edge_r,edge_c])
        h = self.drop(h)
        h = F.relu(h)
        
        h = self.gconv2(g,h,edge_weight=A[edge_r,edge_c])
        h = self.drop(h)
        h = F.relu(h)
         
        h = self.lin1(h)
        h = self.drop(h)
        h = F.relu(h)
        h = self.lin2(h)
         
        return h
    

    