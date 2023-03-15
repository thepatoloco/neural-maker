import unittest
from ..neurons.sigmoid_neuron import SigmoidNeuron

class NeuronsTest(unittest.TestCase):
    def test_neurons_results(self):
        self.assertAlmostEqual(SigmoidNeuron.activation_function(neuron_activation=0), 0.5)
        self.assertAlmostEqual(SigmoidNeuron.activation_function(neuron_activation=100), 1)
        self.assertAlmostEqual(SigmoidNeuron.activation_function(neuron_activation=-100), 0)
