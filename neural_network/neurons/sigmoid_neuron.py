import numpy as np
from .neuron_interface import NeuronInterface

class SigmoidNeuron(NeuronInterface):
    @staticmethod
    def activation_function(neuron_activation: float) -> float:
        """
        activation_function calculates the output of the neuron sigmoid function with the neuron activation
        :param neuron_activation: the neuron activation 
        :return: the output of the neuron
        """
        return (1) / (1 + np.e**(-1 * neuron_activation))

    @staticmethod
    def activation_function_derivative(neuron_activation: float) -> float:
        """
        activation_function_derivative calculates the derivative of the sigmoid function with the neuron activation
        :param neuron_activation: the neuron activation
        :return: the result of the activation function derivative
        """
        return SigmoidNeuron.activation_function(neuron_activation) * (1 - SigmoidNeuron.activation_function(neuron_activation))