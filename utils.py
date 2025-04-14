import numpy as np
from numpy import ndarray


def softmax(x: ndarray) -> ndarray:
    """
    Compute the softmax of a vector x.

    :param x: A NumPy array of shape (n, 1).
    :return: A NumPy array of shape (n, 1) with softmax applied, where the values sum to 1.
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)
