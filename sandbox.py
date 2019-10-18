from NeuronClass import Neuron
from NeuronClass import randomWeights
from activationFunctions import sigmoid
import numpy as np

# create dummy data
input_vector = np.array([2, 4, 11])
input_vector = np.array(input_vector, ndmin=2).T

neuron1 = Neuron("firstNeuron", randomWeights(), sigmoid(x = 2))

neuron1.saveWeights("StartWeights")

# check all the properties of an obj
# attrs = vars(neuron1)






