import numpy as np
import pickle
from typing import Union
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def confusion_matrix(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> None:
    """
    Creates a confusion matrix using a heatmap.
    
    :param labels: A tensor or ndarray of true labels.
    :params preds: A tensor or ndarray of predicted labels.
    """

    cm = (confusion_matrix(preds,labels))
    target_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names,cmap='viridis')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show(block=False)

