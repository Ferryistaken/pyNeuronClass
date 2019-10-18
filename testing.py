import numpy as np
from numpy import exp, array, random, dot, tanh
from enum import Enum
from typing import Union
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Neuron:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

