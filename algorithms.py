import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def propagate(weights, bias, data, labels):

    m = data.shape[1]

    # Forward propagation
    activation = sigmoid(np.dot(weights.T, data) + bias)
    cost = (-1 / m) * np.sum(labels * np.log(activation) + (1 - labels) * np.log(1 - activation))
    
    # Backward propagation
    weight_derivative = (1 / m) * (np.dot(data, (activation - labels).T))
    bias_derivative   = (1 / m) * np.sum(activation - labels)

    cost = np.squeeze(cost)
    grads = {'dw': weight_derivative, 'db': bias_derivative}

    return grads, cost


def optimize(weights, bias, data, labels, niter, lrate, verbose=False):
    costs = []

    for i in range(niter):
        grads, cost = propagate(weights, bias, data, labels)

        weight_derivative = grads['dw']
        bias_derivative   = grads['db']

        weights = weights - lrate * weight_derivative
        bias    = bias    - lrate * bias_derivative

        # Record progress of costs (each 100th)
        if i % 100 == 0:
            costs.append(cost)
            if verbose:
                print(f'i: {i}\tcost: {cost}')
        
    return weights, bias, weight_derivative, bias_derivative, costs


def predict(weights, bias, data):
    m = data.shape[1]
    prediction = np.zeros((1, m))
    weights = weights.reshape(data.shape[0], 1)

    activation = sigmoid(np.dot(weights.T, data) + bias)

    for i in range(activation.shape[1]):
        prediction[0][i] = 0 if activation[0][i] <= 0.5 else 1

    return prediction


def model(data, labels, niter, lrate, verbose):
    weight = np.zeros(shape=(data.shape[0], 1))
    bias   = 0

    weights, bias, weight_derivative, bias_derivative, costs = optimize(weight, bias, data, labels, niter, lrate, verbose)

    prediction = predict(weights, bias, data)
    
    accuracy = 100 - np.mean(np.abs(prediction - labels)) * 100

    if verbose:
        print(f'accuracy: {accuracy}')
    
    return costs, prediction, weights, bias, lrate, niter
    