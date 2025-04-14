import numpy as np
from numpy import ndarray
from input_layer import InputLayer
from hidden_layer import HiddenLayer
from output_layer import OutputLayer
from typing import List


class SimpleRNN:
    """
    A basic implementation of a simple Recurrent Neural Network (RNN) that processes an input sequence and produces an
    output sequence.

    Attributes:
        input_layer (InputLayer): An input layer.
        hidden_layer (HiddenLayer): A hidden layer.
        output_layer (OutputLayer): An output layer.
        hidden_size (int): The number of neurons of the hidden layer.
        learning_rate (float): The learning rate used during weight updates.

    Methods:
        feed_forward(inputs):
            Perform forward propagation to process a sequence of inputs to compute predictions at each time step.
        backpropagation(expected):
            Perform backpropagation through time over the entire sequence in reverse order. Update all gradients and
            weights.
        loss(y_pred, y_true):
            Calculate the cross-entropy loss function.
        train(inputs, expected, epochs):
            Train the model.
    """
    input_layer: InputLayer = None
    hidden_layer: HiddenLayer
    output_layer: OutputLayer
    hidden_size: int
    learning_rate: float

    def __init__(self, time_steps: int, hidden_size: int, learning_rate: float) -> None:
        """
        Initialize a SimpleRNN instance.

        :param time_steps: The number of time steps in the input sequence.
        :param hidden_size: The number of neurons of the hidden layer.
        :param learning_rate: The learning rate.
        """
        self.hidden_layer = HiddenLayer(time_steps, hidden_size)
        self.output_layer = OutputLayer(time_steps, hidden_size)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

    def feed_forward(self, inputs: ndarray) -> OutputLayer:
        """
        Perform forward propagation to process a sequence of inputs to compute predictions at each time step.

        :param inputs: The sequence of inputs.
        :return: The output layer containing predictions for each time step.
        """
        self.input_layer = InputLayer(inputs, self.hidden_size)
        for time_step in range(len(inputs)):
            weighted_input = self.input_layer.weighted_sum(time_step)
            activation = self.hidden_layer.forward(weighted_input, time_step)
            self.output_layer.predict(activation, time_step)
        return self.output_layer

    def backpropagation(self, expected: ndarray) -> None:
        """
        Perform backpropagation through time over the entire sequence in reverse order. Update all gradients and
        weights.

        :param expected: The expected output.
        """
        for time_step in reversed(range(len(expected))):
            delta_output = self.output_layer.calc_gradient(time_step, expected[time_step],
                                                           self.hidden_layer.get_hidden_state(time_step))
            delta_weighted_sum = self.hidden_layer.calc_gradient(time_step, delta_output)
            self.input_layer.calc_gradient(time_step, delta_weighted_sum)

        self.output_layer.update_weights(self.learning_rate)
        self.hidden_layer.update_weights(self.learning_rate)
        self.input_layer.update_weights(self.learning_rate)

    def loss(self, y_pred: List[ndarray], y_true: List[ndarray]) -> float:
        """
        Calculate the cross-entropy loss function.

        :param y_pred: Predicted value.
        :param y_true: Expected value.
        :return: Total loss.
        """
        return sum(np.sum(y_true[i] * np.log(y_pred[i]) for i in range(len(y_true))))

    def train(self, inputs: ndarray, expected: ndarray, epochs: int) -> None:
        """
        Train the model.

        :param inputs: A sequence of inputs.
        :param expected: The expected output values corresponding to the input.
        :param epochs: The number of complete passes through the training data.
        """
        for epoch in range(epochs):
            for idx, inp in enumerate(inputs):
                self.feed_forward(inp)
                self.backpropagation(expected[idx])
