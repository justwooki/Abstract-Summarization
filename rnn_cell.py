import numpy as np
from numpy import ndarray
from typing import Callable, Tuple
from utils import get_activation


class RNNCell:
    W: ndarray  # input-to-hidden weight matrix
    R: ndarray  # hidden-to-hidden (recurrent) weight matrix
    b: ndarray  # bias vector
    dW: ndarray  # gradient of W
    dR: ndarray  # gradient of R
    db: ndarray  # gradient of b
    activation: Callable[[ndarray], ndarray]
    activation_derivative: Callable[[ndarray], ndarray]
    _cache: Tuple[ndarray, ndarray, ndarray]

    def __init__(self, input_dim: int, hidden_dim: int, activation: str) -> None:
        """
        Initialize RNN cell parameters.

        :param input_dim: Dimensionality of input features per time step.
        :param hidden_dim: Number of units in the hidden state.
        :param activation: Name of activation function.
        """
        # activation and its derivative
        self.activation, self.activation_derivative = get_activation(activation)

        # weight matrices and bias
        self.W = np.random.randn(hidden_dim, input_dim) * 0.1
        self.R = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b = np.zeros((hidden_dim, 1))

        # gradients initialization
        self.dW = np.zeros_like(self.W)
        self.dR = np.zeros_like(self.R)
        self.db = np.zeros_like(self.b)

    def forward(self, x_t: ndarray, h_prev: ndarray) -> ndarray:
        """
        Perform one time step forward in this RNN cell.

        :param x_t: Input vector at time step t, shape (input_dim, 1).
        :param h_prev: Previous hidden layer output, shape (hidden_dim, 1).
        :return: Current hidden layer output, shape (hidden_dim, 1).
        """
        z = self.W.dot(x_t) + self.R.dot(h_prev) + self.b
        h_t = self.activation(z)
        self._cache = (x_t, h_prev, z)  # save intermediate values for backprop
        return h_t

    def backprop(self, d_h: ndarray) -> ndarray:
        """
        Backpropagate through one time step of this RNN cell.

        :param d_h: Gradient of loss L with respect to the current hidden layer output, shape (hidden_dim, 1).
        :return: Gradient of loss L with respect to the previous hidden layer output, shape (hidden_dim, 1).
        """
        x_t, h_prev, z = self._cache
        dz = d_h * self.activation_derivative(z)
        self.dW += dz.dot(x_t.T)
        self.dR += dz.dot(h_prev.T)
        self.db += dz
        dh_prev = self.R.T.dot(dz)
        return dh_prev

    def update_weights(self, lr: float) -> None:
        """
        Update RNN cell weights using accumulated gradients.

        :param lr: Learning rate.
        """
        self.W -= lr * self.dW
        self.R -= lr * self.dR
        self.b -= lr * self.db

        self.dW.fill(0)
        self.dR.fill(0)
        self.db.fill(0)
