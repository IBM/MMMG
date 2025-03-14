import matplotlib.pyplot as plt
plt.close('all')

#import deep learning tools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class feature_transformer(torch.nn.Module):
    
    """
    transformer run on concatenated input features with MHA
    
    Inputs:
        in_feats: input features
        h_feats: size of encoding
        num_classes: no of outcomes
        
    Methods:
        forward: forward pass
            Inputs:
                x: input vector
        train: trainer for transformer
            
            Inputs:
                concat_inputs: dictionary [train,val,test] of concatenated reduced modality features
               
    """
    
    def __init__(self,in_feats,num_heads, h_feats,  num_classes):
        super(feature_transformer, self).__init__()
        
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(in_feats,h_feats)
        
        self.te1 = nn.TransformerEncoderLayer(1,num_heads,batch_first=True,dim_feedforward=1)
        self.transformer_encoder = nn.TransformerEncoder(self.te1, num_layers=2)
        
        self.fc2 = nn.Linear(h_feats,num_classes)
        
    def forward(self,x):
        
        [m,n] = x.shape
        
        x = self.transformer_encoder(x.view(m,n,1)) 
        
        output = F.relu(self.fc1(x.reshape(m,n)))
        output = self.fc2(output)
        
        
        return output
    
    def trainer(self,concat_inputs,targets,lr,num_epochs):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr,betas=(0.9,0.999),eps=1e-08,weight_decay=0.5)
        best_val_acc = 0
        best_test_acc = 0

        train_feats = concat_inputs[0]
        val_feats = concat_inputs[1]
        test_feats = concat_inputs[2]
    
        targets_train = torch.tensor(targets[0],dtype=torch.int64)
        targets_val = torch.tensor(targets[1],dtype=torch.int64)
        targets_test = torch.tensor(targets[2],dtype=torch.int64)

        N = targets_train.shape[0]
        batch_size = 128
    
        num_loop = int(np.ceil(N/batch_size))
        ptation = np.random.permutation(np.arange(N))
    
    
        for e in range(num_epochs):
        
            # Forward pass
            running_loss = 0
            for j in range(num_loop):
            
                print("Running batch %d of %d" %(j+1,num_loop))
            
                if j== num_loop-1:
                    indices = ptation[j*batch_size:].astype(int)
                    eff_bs = len(indices)
                
                else:
                
                    indices = ptation[j*batch_size:(j+1)*batch_size].astype(int)
                    eff_bs = batch_size
                     
            
                self.train()
            
                datainputs_batch = train_feats[indices,:]
            
                targets_train_batch = targets_train[indices]
                logits_train_batch = self.forward(datainputs_batch)
            
            
                loss = F.cross_entropy(logits_train_batch, targets_train_batch)
            
                #backward pass
                optimizer.zero_grad()
            
                loss.backward()
                optimizer.step()
       
            
                running_loss +=loss.item()
                print("Epoch %d done, running validation stats ... running loss %1.3f" %(e,running_loss/num_loop))
            
                del datainputs_batch #save memory
       
            logits_train = self.forward(train_feats)
            pred_train = logits_train.argmax(1)
       

            # Compute accuracy on training/validation/test
            with torch.no_grad():
            
                logits_val = self.forward(val_feats)
                logits_test = self.forward(test_feats)
            
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
         