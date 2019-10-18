from NeuronClass import Neuron
from NeuronClass import randomWeights
from activationFunctions import sigmoid
import numpy as np

neuron1 = Neuron("firstNeuron", randomWeights(), sigmoid(x = 1))