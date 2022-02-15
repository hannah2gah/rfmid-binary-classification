# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

def test_model(y_test, yhat):
    '''
    Compute accuracy, precision, recall, f1_score, and confusion_matrix

    Parameters
    ----------
    y_test : TYPE
        DESCRIPTION.
    yhat : TYPE
        DESCRIPTION.

    Returns
    -------
    model_acc : TYPE float
        DESCRIPTION.
    model_rec : TYPE float
        DESCRIPTION.
    model_prec : TYPE float
        DESCRIPTION.
    model_f1 : TYPE float
        DESCRIPTION.
    model_cm : TYPE float
        DESCRIPTION.

    '''
    # evaluate model prediction
    model_acc = accuracy_score(y_test, yhat)
    model_prec = precision_score(y_test, yhat)
    model_rec = recall_score(y_test, yhat)
    model_f1 = f1_score(y_test, yhat)
    model_cm = confusion_matrix(y_test, yhat)
    
    print('Accuracy score: ',  np.round((model_acc*100),3))
    print('Precision score: ', np.round((model_prec*100),3))
    print('Recall score: ', np.round((model_rec*100),3))
    print('F1 score: ',  np.round((model_f1*100),3))
    
    ax = sns.heatmap(model_cm, annot=True, fmt='d')
    ax.set_xlabel('Predicted Value')
    ax.set_ylabel('Actual Value')
    return model_acc, model_prec, model_rec, model_f1, model_cm
#%%

def plot_roc_curve(class_labels, test_labels):
    '''
    Plot ROC Cruve of a model

    Parameters
    ----------
    class_labels : TYPE array
        DESCRIPTION. 
    test_labels : TYPE array
        DESCRIPTION. prediction labels

    Returns
    -------
    fpr : TYPE array
        DESCRIPTION.
    tpr : TYPE array
        DESCRIPTION.
    auc : TYPE float
        DESCRIPTION.

    '''
    
    fpr, tpr, _ = roc_curve(class_labels, test_labels)
    auroc = auc(fpr,tpr)

    # Plot comparison
    plt.style.use('seaborn')
    lw = 2

    # Plot ROC Curve
    plt.plot(fpr, tpr, color='indigo',
         lw=lw, label='ROC area = %0.3f' % auroc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    return fpr, tpr, auc
