import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, roc_auc_score

def draw_loss(train_loss_list, val_loss_list):
    plt.figure()
    plt.plot(np.arange(len(train_loss_list)), train_loss_list, label = 'train loss')
    plt.plot(np.arange(len(val_loss_list)), val_loss_list, label = 'val loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()

def draw_auc_epoch(auc_list):
    plt.figure()
    plt.plot(np.arange(len(auc_list)), auc_list, label = 'auc_epoch')
    plt.title('auc_epoch')
    plt.legend()
    plt.savefig('auc.png')
    plt.close()

def compute_auc(labels, outputs):
    fpr, tpr, thresholds = roc_curve(labels, outputs)
    auc_val = auc(fpr, tpr)
    return auc_val