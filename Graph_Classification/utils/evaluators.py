#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 23:12:18 2022

@author: niharika.dsouza
"""

from sklearn.metrics import roc_curve, auc     
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
plt.close('all')


def performance_evaluate(y_true, y_pred, title_str, folder_name):
    
    """
    Evaluate class wise metrics: auc 
    
    Inputs: 
        y_true: True labels
        y_pred: Predicted labels
        title_str: method name
        folder_name: name of folder to save results in
    """
    
    
    fpr = dict()
    tpr = dict()    
    roc_auc = dict()
    
    y_pred = y_pred.detach().numpy()
    y_true = label_binarize(y_true, classes=[0,1,2,3,4])
    n_classes = y_true.shape[1]
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i] )
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        lw = 2
        plt.plot(fpr[i], tpr[i], color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title_str + str(i))
        plt.legend(loc="lower right")
        plt.savefig(folder_name+ "/roc_" + title_str + str(i) + ".png")
#         plt.show()   
        
    return roc_auc