import numpy as np
from numpy import ndarray
from typing import List
from rnn_cell import RNNCell

class RNN:
    E: ndarray  # embedding matrix (vocab size, embeddings dimension)
    cells: List[RNNCell]
    lr: float

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dims: List[int], activations: List[str],
                 learning_rate: float) -> None:
        """
        Create a multi-layer RNN.

        :param vocab_size: Number of tokens in vocabulary.
        :param embed_dim: Dimensionality of token embeddings.
        :param hidden_dims: List of sizes of hidden layers.
        :param activations: Activation names for each RNNCell.
        :param learning_rate: Learning rate for gradient updates.
        """
        assert len(activations) == len(hidden_dims), "Activations must match hidden layers"

        self.E = np.random.randn(vocab_size, embed_dim) * 0.1
        layer_dims = [embed_dim] + hidden_dims
        self.cells = [
            RNNCell(
                input_dim=layer_dims[i],
                hidden_dim=layer_dims[i + 1],
                activation=activations[i]
            ) for i in range(len(layer_dims) - 1)
        ]
        self.lr = learning_rate

    def predict(self, inputs: ndarray) -> ndarray:
        """
        Generate predictions by running forward pass.

        :param inputs: The sequence of inputs.
        :return: The final hidden state vector.
        """
        layer_outputs = [np.zeros((cell.R.shape[0], 1)) for cell in self.cells]
        embedded_input = self.E[inputs]

        for time_step in range(embedded_input.shape[0]):
            x_t = embedded_input[time_step]
            for idx, cell in enumerate(self.cells):
                h = cell.forward(x_t, layer_outputs[idx])
                layer_outputs[idx] = h
                x_t = h

        return layer_outputs[-1]

    def loss(self, y_pred: List[ndarray], y_true: List[ndarray]) -> float:
        """
        Calculate the cross-entropy loss function.

        :param y_pred: Predicted value.
        :param y_true: Expected value.
        :return: Total loss.
        """
        return sum(np.sum(y_true[i] * np.log(y_pred[i]) for i in range(len(y_true))))

    def backprop(self, d_o: ndarray) -> None:
        """
        Backpropagate through time and layers.

        :param d_o: Gradient of loss L with respect to the final output layer, shape (hidden_dim, 1).
        """
        d_h = d_o
        for cell in reversed(self.cells):
            d_h = cell.backprop(d_h)

    def update_weights(self) -> None:
        """
        Update all RNN cell weights using their accumulated gradients.
        """
        for cell in self.cells:
            cell.update_weights(self.lr)

    def fit(self, X: ndarray, y: ndarray, epochs: int) -> None:
        """
        Train the RNN model.

        :param X: 2D array of token ID sequences.
        :param y:
        :param epochs: Number of epochs to train.
        """
        for epoch in range(epochs):
            for inputs, y_true in zip(X, y):
                y_pred = self.predict(inputs)
                d_o = y_pred - y_true
                self.backprop(d_o)
                self.update_weights()
