import numpy as np
from numpy import ndarray


class InputLayer:
    """
    Represents an input layer in a neural network.

    Attributes:
        inputs (ndarray): Sequential data in the form of numpy arrays. Each entry represents an input vector for a
                          particular time step.
        U (ndarray): Weight matrix connecting the input to the hidden layer.
        delta_U (ndarray): Gradient of U calculated during BPTT (backpropagation through time).

    Methods:
        get_inputs(time_step):
            Retrieve the input vector at a specific time step.
        weighted_sum(time_step):
            Compute the weighted sum for the input at the given time step.
        calc_gradient(time_step, delta_weighted_sum):
            Calculate the gradient for the weights for the given time step.
        update_weights(learning_rate):
            Update the weights using the gradient.
    """
    inputs: ndarray
    U: ndarray = None
    delta_U: ndarray = None

    def __init__(self, inputs: ndarray, hidden_size: int) -> None:
        """
        Initialize the InputLayer with the provided input data and desired hidden layer size.

        :param inputs: The input data; typically a 2D array where each entry represents an input vector.
        :param hidden_size: The number of neurons (units) in the hidden representation this layer will transform the
                            input into.
        """
        self.inputs = inputs
        self.U = np.random.uniform(low=0, high=1, size=(hidden_size, len(inputs[0])))
        self.delta_U = np.zeros_like(self.U)

    def get_inputs(self, time_step: int) -> ndarray:
        """
        Retrieve the input vector at a specific time step.

        :param time_step: The time index from which to retrieve the input vector.
        :return: The input vector at the specified time step.
        """
        return self.inputs[time_step]

    def weighted_sum(self, time_step: int) -> ndarray:
        """
        Compute the weighted sum for the input at the given time step.

        :param time_step: The time index for which to compute the weighted sum.
        :return: The resulting weighted sum.
        """
        return self.U.dot(self.get_inputs(time_step))

    def calc_gradient(self, time_step: int, delta_weighted_sum: ndarray) -> None:
        """
        Calculate the gradient for U for the given time step.

        :param time_step: The current time step.
        :param delta_weighted_sum: The gradient with respect to the weighted sum. Expected shape is (hidden_size, 1).
        """
        # (h_dim, 1) * (1, input_size) = (h_dim, input_size)
        self.delta_U += delta_weighted_sum.dot(self.get_inputs(time_step).T)

    def update_weights(self, learning_rate: float) -> None:
        """
        Update the weights using the gradient.

        :param learning_rate: The learning rate used to scale the weight updates.
        """
        self.U -= learning_rate * self.delta_U
