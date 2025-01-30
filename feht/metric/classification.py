import numpy as np
import torch


def accuracy(
        y_true: list|np.ndarray|torch.Tensor, 
        y_pred: list|np.ndarray|torch.Tensor
        ) -> float:
    '''
    Calculate the accuracy.

    :param y_true: true labels
    :type y_true: list|np.ndarray|torch.Tensor
    :param y_pred: predicted labels
    :type y_pred: list|np.ndarray|torch.Tensor

    :return: accuracy
    :rtype: float
    '''
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    
    assert y_true.shape == y_pred.shape, "y_true and y_pred should have the same shape."
    assert len(y_true.shape) == len(y_pred.shape) == 1, "y_true and y_pred should be 1D array."

    return np.mean(y_true == y_pred)