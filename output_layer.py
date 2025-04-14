import numpy as np
from numpy import ndarray
from utils import softmax


class OutputLayer:
    """
    Represents an output layer in a neural network.

    Attributes:
        states (ndarray): Stores predictions of all time steps (internal memory of the network).
        V (ndarray): Output weight matrix that maps hidden state to the output.
        delta_V (ndarray): Gradient of V calculated during BPTT (backpropagation through time).
        bias (ndarray): The bias term.
        delta_bias (ndarray): Gradient of the bias term calculated during BPTT.

    Methods:
        get_state(time_step):
            Retrieves the output state (prediction) value at a given time step.
        set_state(time_step, prediction):
            Update the output state (prediction) value at a given time step after prediction calculation.
        prediction(hidden_state, time_step):
            Calculates output prediction at a given time step.
        calc_gradient(time_step, expected, hidden_state): Calculate the gradient for the current activation function
                                                          with respect to the current output and update the gradient
                                                          for V and the bias term.
        update_weights(learning_rate):
            Update the weights and bias using the gradient.
    """
    states: ndarray = None
    V: ndarray = None
    delta_V: ndarray = None
    bias: ndarray = None
    delta_bias: ndarray = None

    def __init__(self, size: int, hidden_size: int) -> None:
        """
        Initializes OutputLayer with number of output units and hidden units coming to the output layer.

        :param size: The number of output neurons (units).
        :param hidden_size: The number of neurons coming from the hidden layer to the output layer.
        """
        self.states = np.zeros(shape=(size, size, 1))
        self.V = np.random.uniform(low=0, high=1, size=(size, hidden_size))
        self.delta_V = np.zeros_like(self.V)
        self.bias = np.random.uniform(low=0, high=1, size=(size, 1))
        self.delta_bias = np.zeros_like(self.bias)

    def get_state(self, time_step: int) -> ndarray:
        """
        Retrieves the output state (prediction) value at a given time step.

        :param time_step: The time step at which to retrieve the prediction value.
        :return: The prediction value at the given time step.
        """
        return self.states[time_step]

    def set_state(self, time_step: int, prediction: ndarray) -> None:
        """
        Update the output state (prediction) value at a given time step after prediction calculation.

        :param time_step: The time step at which to set the prediction value.
        :param prediction: The updated prediction value to set.
        """
        self.states[time_step] = prediction

    def predict(self, hidden_state: ndarray, time_step: int) -> ndarray:
        """
        Calculates output prediction at a given time step.

        :param hidden_state: The activations from the preceding hidden layer.
        :param time_step: The current time step.
        :return: The output prediction.
        """
        # V * h => (output_size, h_dim) * (h_dim, 1) = (output_size, 1)
        # (output_size, 1) + (output_size, 1) = (output_size, 1)
        output = self.V.dot(hidden_state) + self.bias
        prediction = softmax(output)
        self.states[time_step] = prediction
        return prediction

    def calc_gradient(self, time_step: int, expected: ndarray, hidden_state: ndarray) -> ndarray:
        """
        Calculate the gradient for the current activation function with respect to the current output and update the
        gradient for V and the bias term.

        :param time_step: The current time step for which the gradient is being calculated.
        :param expected: The expected output.
        :param hidden_state: The current hidden state activation value.
        :return: The gradient for the current activation function with respect to the current output.
        """
        # (output_size, 1) - (output_size, 1) = (output_size, 1)
        delta_output = self.get_state(time_step) - expected
        # (output_size, 1) * (1, h_dim) = (output_size, h_dim)
        self.delta_V += delta_output.dot(hidden_state.T)
        self.delta_bias += delta_output
        # (h_dim, output_size) * (output_size, 1) = (h_dim, 1)
        return self.V.T.dot(delta_output)

    def update_weights(self, learning_rate: float) -> None:
        """
        Update the weights and bias using the gradient.

        :param learning_rate: The learning rate used to scale the weight and bias updates.
        """
        self.V -= learning_rate * self.delta_V
        self.bias -= learning_rate * self.delta_bias
