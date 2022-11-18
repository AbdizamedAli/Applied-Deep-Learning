
def confusion_matrix(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray] class_size = 10: int) -> :
    '''
    Generate a confusion matrix
    '''

    confusion_matrix = torch.zeros(class_size,class_size)

