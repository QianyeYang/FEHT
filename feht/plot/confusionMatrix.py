import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
        y_true: list[int]|np.ndarray,
        y_pred: list[int]|np.ndarray,
        target_names: list[str],
        title: str='Confusion matrix',
        cmap: str|None=None,
        normalize: bool=False,
        save_path: str='',
        ) -> None:
    """
    Plot, or save a confusion matrix.

    :param y_true: true labels
    :type y_true: list|np.ndarray
    :param y_pred: predicted labels
    :type y_pred: list|np.ndarray
    :param target_names: target names
    :type target_names: list
    :param title: title of the plot, defaults to 'Confusion matrix'
    :type title: str
    :param cmap: color map, defaults to None
    :type cmap: str
    :param normalize: whether to normalize the confusion matrix, defaults to False
    :type normalize: bool
    :param save: path to save the confusion matrix, defaults to ''
    :type save: str

    :return: None
    """
    if isinstance(y_true, np.ndarray):
        assert y_true.ndim == 1, "y_true should be 1D array..."
        y_true = y_true.tolist()

    if isinstance(y_pred, np.ndarray):
        assert y_pred.ndim == 1, "y_pred should be 1D array..."
        y_pred = y_pred.tolist()

    assert len(y_true) == len(y_pred), "y_true and y_pred should have the same length..."
    # assert len(target_names) == len(np.unique(y_true)), \
    #     "target_names should have the same length as the unique values in y_true..."
    
    cm = confusion_matrix(y_true, y_pred)

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.3f}; misclass={:0.3f}'.format(accuracy, misclass))

    if save_path == '':
        plt.show()
    else:
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':

    y_true = [0, 1, 2, 2, 1, 0]
    y_pred = [0, 1, 2, 1, 0, 0]

    plot_confusion_matrix(
        y_true = y_true,
        y_pred = y_pred,
        target_names = ['Class 0', 'Class 1', 'Class 2'],
        title = "Confusion Matrix",
        save_path='confusion_matrix.png'
        )
    
    