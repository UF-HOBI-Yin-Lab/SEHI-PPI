
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

def sensitivity(y_true, y_pred):
    y_true = np.array(y_true)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)

def specificity(y_true, y_pred):
    y_true = np.array(y_true)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def auc(y_true, y_pred, average='macro'):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return metrics.roc_auc_score(y_true, y_pred, average=average)

def mcc(y_true, y_pred):
    y_true = np.array(y_true)
    return metrics.matthews_corrcoef(y_true, y_pred)

def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    return metrics.accuracy_score(y_true, y_pred)

def f1(y_true, y_pred, average='macro'):
    y_true = np.array(y_true)
    return metrics.f1_score(y_true,y_pred, average=average, zero_division=0)

def cofusion_matrix(y_true,y_pred):
    y_true = np.array(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp