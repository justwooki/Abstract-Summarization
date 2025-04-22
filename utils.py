import numpy as np
from numpy import ndarray
from typing import Callable, Tuple


def tanh(x: ndarray) -> ndarray:
    """
    Hyperbolic tangent activation for hidden-state updates.

    :param x: Input array.
    :return: Output array where each element is tanh of the corresponding input.
    """
    return np.tanh(x)


def tanh_derivative(x: ndarray) -> ndarray:
    """
    Derivative of tanh activation.

    :param x: Input array.
    :return: Output array that is the derivative of the tanh activation.
    """
    y = np.tanh(x)
    return 1.0 - y * y


def softmax(x: ndarray) -> ndarray:
    """
    Softmax activation for output layers, producing a probability distribution.

    :param x: Input array of shape (n, 1): logits for each class.
    :return: Output array of same shape with softmax probabilities that sum to 1.
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def softmax_derivative(x: ndarray) -> ndarray:
    """
    Derivative of softmax activation.

    :param x: Logits array of shape (n, 1).
    :return: Array of same shape that is the derivative of the softmax activation.
    """
    s = softmax(x)
    return s * (1 - s)


def get_activation(name: str) -> Tuple[Callable[[ndarray], ndarray], Callable[[ndarray], ndarray]]:
    """
    Retrieve activation function and its derivative by name.

    :param name: Name of the activation.
    :return: Callable that applies the corresponding activation and derivative of the activation.
    :raises ValueError: If the activation is not supported.
    """
    name = name.lower()

    if name == 'tanh':
        return tanh, tanh_derivative
    elif name == 'softmax':
        return softmax, softmax_derivative
    else:
        raise ValueError(f"Unsupported activation function '{name}'")
