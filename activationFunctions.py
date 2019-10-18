import math
import matplotlib.pyplot as plt

# sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# rectified linear unit
def ReLU(x):
	# rectified linear function
	return max(0.0, x)



