import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_predClamp = np.clip(y_pred, 1e-7,1 - 1e-7)
        y1 = y_true * np.log(y_predClamp)
        y2 = (1-y_true) * np.log(1-y_predClamp)
        y2+=y1
        y3 = -np.mean(y2)
        return round(y3,4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred:NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        clipped = np.clip(y_pred,1e-7,1-1e-7)
        y1 = y_true * np.log(clipped)
        y2 = np.sum(y1,axis=1)
        y2 = - np.mean(y2)
        return round(y2,4)









