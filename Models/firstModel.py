from NeuronClass import Neuron
from NeuronClass import randomWeights
from activationFunctions import sigmoid
import numpy as np

i = [1, 2, 3, 4, 5]
n1 = Neuron("N1", [0, 1, 1, 0, 0], sigmoid)
n2 = Neuron("N2", [0, 1, 0, 1, 0], sigmoid)
n3 = Neuron("N3", [1, 1, 1, 0, 0], sigmoid)
n4 = Neuron("N4", [0, 1, 0, 1, 1], sigmoid)
n5 = Neuron("N5", [1, 0, 0, 0, 1], sigmoid)

l1=[n1,n2,n3,n4]
o1=[0, 0, 0, 0]

n10 = Neuron("N10", [1, 1, 0, 0], sigmoid)
n11 = Neuron("N11", [1, 1, 1, 0], sigmoid)
n12 = Neuron("N12", [0, 0, 1, 1], sigmoid)

l2=[n10,n11,n12]


for n in l1:
	_i=0
	n.computeOutput(i)
	o1[_i]=n.getOutput()
	_i=_i+1

for n in l2:
	n.computeOutput(o1)

for n in l2:
	print(n.getOutput())