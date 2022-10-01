# a module to get some of the most common activation functions and their derivatives if it's appropriate.
# the inputs are intended to be numpy arrays which take advantage of array bradcasting.
import numpy as np


def ReLU(Z):
    return np.maximum(Z,0)

def derivReLU(Z):
    return Z > 0

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def derivSigmoid(Z):
    return sigmoid(Z) * (1 - sigmoid(Z)) # commonly known expression for the derivative

def softMax(Z):
    return np.exp(Z)/(sum(np.exp(Z)))

# I don't need to use this as  I one use softmax on the output layer so I won't implement it here since it appears to be complicated to work out
# def deriv_softMax(Z):

def tanh(Z):
    return np.tanh(Z)

def derivTanh(Z):
    return 1 - np.square(np.tanh(Z))