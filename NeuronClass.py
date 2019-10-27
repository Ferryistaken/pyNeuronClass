import numpy as np
from numpy import exp, array, random, dot, tanh
from configparser import ConfigParser
from errorCorrection.gradientDescent import firstGradient

# random weight between 0 and 1
def randomWeights(n):
    # create an array of floats w/ minimun value of -1, max of 1 and of size 5
    weightArray = np.random.uniform(low = -1, high = 1, size = (1, n))

    # i might find a use for this later
    randWeights = random.uniform(0, 1)
    return weightArray

# this is to test if other values are array of floats
#testArray = np.array(randomWeights())


class Neuron:
    def __init__(self, id, weights, f):
#        if type(weights) == type(testArray):
        self._weights = weights
#        else:
#            print("The weights you entered aren't a vector of floats. What you entered : " + str(weights) + " The data type of your values = " +
#                  str(type(weights)))
        self._f = f
        self.id = id

    # weights are 0 because they need to be overwritten
    _weights = 0
    _inputs = 0
    _output = 0

    # def an identity function (input = output)
    def _identity(self, input):
        return input

    # assigning the identity function to the var _f
    _f = _identity

    # computing output regardless of knowing the actual output
    def computeOutput(self, inputs):
        self._output = self._f(np.dot(inputs, self._weights))
        self._inputs = inputs
        return(0)

    def getOutput(self):
        return(self._output)

    # during training it sees if the output is different from the input("training function")
    def feedForward(self, inputs, trainOutputs):
        self._inputs = inputs
        self.computeOutput(inputs)

        return self._output

    # function to save the weights
    def saveWeights(self, name):
            try:
                weightConfig = ConfigParser()
                weightConfig.read(name + '.ini')
                weightConfig.add_section('Weights')
                weightConfig.set('Weights', 'weight1', str(self._weights))

                with open(name + '.ini', 'w') as f:
                    weightConfig.write(f)
            except:
                print("error while saving weights. The current weights are : " + self._weights)


