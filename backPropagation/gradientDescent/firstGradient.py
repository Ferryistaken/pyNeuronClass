

def step_gradient(x, y, weight, bias, learning_rate):
    temp_weight = 0
    temp_bias = 0

    N = float(x.shape[0])

    for j in range(x.shape[0]):
        temp_bias += - (2 / N) * (y[j] - ((weight * x[j]) + bias))
        temp_weight += -(2 / N) * x[j] * (y[j] - ((weight * x[j]) + bias))

    new_weight = weight - (learning_rate * temp_weight)
    new_bias = bias - (learning_rate * temp_bias)

    return new_weight, new_bias


def gradient_descent(x, y, bias, weight, learning_rate, iterations):
    for i in range(0, iterations):
        new_weight, new_bias = step_gradient(x, y, weight, bias, learning_rate)

        weight = new_weight
        bias = new_bias

    return weight, bias
