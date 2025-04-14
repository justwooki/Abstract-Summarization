import numpy as np
from numpy import ndarray


class HiddenLayer:
    """
    Represents a hidden layer in a neural network.

    Attributes:
        states (ndarray): Stores activation of all time steps (internal memory of the network).
        W (ndarray): Recurrent weight matrix connecting the previous hidden state to this hidden state.
        delta_W (ndarray): Gradient of W calculated during BPTT (backpropagation through time).
        bias (ndarray): The bias term.
        delta_bias (ndarray): Gradient of the bias term calculated during BPTT.
        next_delta_activation (ndarray): Stores the derivative of next step’s loss function with respect to the current
                                         activation.

    Methods:
        get_hidden_state(time_step):
            Retrieve hidden state value at a given time step.
        set_hidden_state(time_step, hidden_state):
            Update the hidden state at a given time step after forward pass calculation.
        forward(weighted_input, time_step):
            Compute and update the hidden state for a given time step.
        calc_gradient(time_step, delta_output):
            Calculate the gradient for the weighted sum of the given time step and update the gradient for W and the
            bias term.
        update_weights(learning_rate):
            Update the weights and bias using the gradient.
    """
    states: ndarray = None
    W: ndarray = None
    delta_W: ndarray = None
    bias: ndarray = None
    delta_bias: ndarray = None
    next_delta_activation: ndarray = None

    def __init__(self, time_steps: int, size: int) -> None:
        """
        Initialize the HiddenLayer with the given number of time steps and hidden state size.

        :param time_steps: The number of time steps in the input sequence.
        :param size: The number of neurons (units) in the hidden layer.
        """
        self.states = np.zeros(shape=(time_steps, size, 1))
        self.W = np.random.uniform(low=0, high=1, size=(size, size))
        self.delta_W = np.zeros_like(self.W)
        self.bias = np.random.uniform(low=0, high=1, size=(size, 1))
        self.delta_bias = np.zeros_like(self.bias)
        self.next_delta_activation = np.zeros(shape=(size, 1))

    def get_hidden_state(self, time_step: int) -> ndarray:
        """
        Retrieve hidden state value at a given time step. If given time step is less than zero, default to an all
        zeroes matrix.

        :param time_step: The time step at which to retrieve the hidden state.
        :return: The hidden state at the specified time step.
        """
        if time_step < 0:
            return np.zeros_like(self.states[0])
        return self.states[time_step]

    def set_hidden_state(self, time_step: int, hidden_state: ndarray) -> None:
        """
        Update the hidden state at a given time step after forward pass calculation.

        :param time_step: The time step at which to set the hidden state.
        :param hidden_state: The updated hidden state value to set.
        """
        self.states[time_step] = hidden_state

    def forward(self, weighted_input: ndarray, time_step: int) -> ndarray:
        """
        Compute and update the hidden state for a given time step.

        :param weighted_input: The weighted input from the preceding layer.
        :param time_step: The current time step.
        :return: The activated hidden state for the given time step.
        """
        prev_hidden_state = self.get_hidden_state(time_step - 1)
        # W * h_prev => (h_dim, h_dim) * (h_dim, 1) = (h_dim, 1)
        weighted_hidden_state = self.W.dot(prev_hidden_state)
        # (h_dim, 1) + (h_dim, 1) + (h_dim, 1) = (h_dim, 1)
        weighted_sum = weighted_hidden_state + weighted_input + self.bias
        activation = np.tanh(weighted_sum) # (h_dim, 1)
        self.set_hidden_state(time_step, activation)
        return activation

    def calc_gradient(self, time_step: int, delta_output: ndarray) -> ndarray:
        """
        Calculate the gradient for the weighted sum of the given time step and update the gradient for W and the bias
        term.

        :param time_step: The current time step for which the gradient is being calculated.
        :param delta_output: Gradient of the activation at the given time step with respect to the current output.
        :return: The gradient for the weighted sum of the given time step.
        """
        # (h_dim, 1) + (h_dim, 1) = (h_dim, 1)
        delta_activation = delta_output + self.next_delta_activation
        # scalar * (h_dim, 1) = (h_dim, 1)
        delta_weighted_sum = (1 - self.get_hidden_state(time_step) ** 2) * delta_activation
        # (h_dim, h_dim) * (h_dim, 1) = (h_dim, 1)
        self.next_delta_activation = self.W.T.dot(delta_weighted_sum)
        # (h_dim, 1) * (1, h_dim) = (h_dim, h_dim)
        self.delta_W += delta_weighted_sum.dot(self.get_hidden_state(time_step - 1).T)
        self.delta_bias += delta_weighted_sum
        return delta_weighted_sum

    def update_weights(self, learning_rate: float) -> None:
        """
        Update the weights and bias using the gradient.

        :param learning_rate: The learning rate used to scale the weight and bias updates.
        """
        self.W -= learning_rate * self.delta_W
        self.bias -= learning_rate * self.delta_bias
