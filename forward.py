#!/usr/bin/env python

import numpy as np
from activation import *


def forward(Theta, X, active):
    N = X.shape[0]

    # Add the bias column
    X_ = np.concatenate((np.ones((N, 1)), X), 1)

    # Multiply by the weights
    z = np.dot(X_, Theta.T)

    # Apply the activation function
    a = active.f(z)

    return a


def predict(model, X):
    h = X.copy()

    for i in range(0, len(model), 2):
        theta = model[i]
        activation = model[i+1]

        h = forward(theta, h, activation)

    return np.argmax(h, 1)


def accuracy(y_, y):
    return np.mean((y_ == y.flatten()))*100.


if __name__ == "__main__":
    Theta1 = np.load('input/Theta1.npy')
    Theta2 = np.load('input/Theta2.npy')

    X = np.load('input/X_train.npy')
    y = np.load('input/y_train.npy')

    model = []

    model.append(Theta1)
    model.append(Sigmoid)
    model.append(Theta2)
    model.append(Sigmoid)

    y_ = predict(model, X)

    print(accuracy(y_, y))
