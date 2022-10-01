# the intent is to create a class that organizes the sequenrtial layers of a multilayer perceptron which gets inputs of size n.

import pickle, io
import numpy as np

# initialize a weight matrix of some size
def initializeWeights(n,m):
    return np.random.rand(n,m) - 0.5

def oneHot(Y):
    oneHotY = np.zeros((Y.size, Y.max() + 1))
    oneHotY[np.arange(Y.size), Y] = 1
    oneHotY = oneHotY.T
    return oneHotY

class Network:
    def __init__(self, inputSize, numClasses, learningRate, layers = None):
        if layers:
            self.layersnumInputs = layers
        else:
            self.layers = []
        self.inputSize = inputSize
        self.numClasses = numClasses
        self.learningRate = learningRate
    def addLayer(self, layer):
        self.layers.append(layer)
    def forward(self, X): # could also call this predict since users arew given access to this.
        inp = X
        Zs = []
        As = []
        for lay in self.layers:
            # print(lay, lay.W.shape, inp.shape, lay.b.shape)
            Z = lay.W.dot(inp) + lay.b
            A = lay.activationFunc(Z)
            inp = A
            Zs.append(Z)
            As.append(A)
        return Zs, As
    def backprop(self, Zs, As, X, Y): # assuming Y is one hot encoded
        reverse_dWs = []
        reverse_dbs = []
        # this is the updates for the output layer
        dZ = As[-1] - Y
        _, m = As[-1].shape # m is the number of samples
        dW = (1/m) * dZ.dot(As[-1].T)
        db = (1/m) * np.sum(dZ)
        reverse_dWs.append(dW)
        reverse_dbs.append(db)
        for i in range(len(self.layers) - 2, 0, -1): # we do the back prop for the remaining layers in reverse order (except the input layer)
            dZ = np.multiply(self.layers[i-1].W.T.dot(dZ), self.layers[i].activationDeriv(Zs[i]))
            dW = (1/m) * dZ.dot(As[i-1].T)
            db = (1/m) * np.sum(dZ)
            reverse_dWs.append(dW)
            reverse_dbs.append(db)
            
        dZ = np.multiply(self.layers[1].W.T.dot(dZ), self.layers[0].activationDeriv(Zs[0]))
        dW = (1/m) * dZ.dot(X.T)
        db = (1/m) * np.sum(dZ)
        reverse_dWs.append(dW)
        reverse_dbs.append(db)

        return reverse_dWs[::-1], reverse_dbs[::-1]
    
    def updateParameters(self, dWs, dbs):
        for lay,dW,db in zip(self.layers, dWs, dbs):
            lay.W = lay.W - self.learningRate * dW
            lay.b = lay.b - self.learningRate * db
        
    def fit(self, X, Y, epochs):
        for i in range(epochs):
            Zs, As = self.forward(X)
            dWs, dbs = self.backprop(Zs, As, X, oneHot(Y))
            # for i,dW in enumerate(dWs):
            #     print(i, " ", dW.shape)
            self.updateParameters(dWs,dbs)
            if i % 50 == 0:
                print("Epoch", i, ": ")
                predictions = np.argmax(As[-1],0)
                accuracy = np.sum(predictions == Y) / Y.size
                print(accuracy)
        predictions = np.argmax(As[-1],0)
        accuracy = np.sum(predictions == Y) / Y.size
        return predictions, accuracy

    def save(self,filename):
        pass

class Layer:
    def __init__(self, numNeurons, inputSize, activationFunc, activationDeriv = None, initialW = None, initialb = None):
        self.numNeurons = numNeurons
        self.inputSize = inputSize
        self.activationFunc = activationFunc
        self.activationDeriv = activationDeriv # I allow you to omit the activation derivative if this is the final layer since that won't be used in backprop.
        if initialW:
            self.W = initialW
        else:
            self.W = initializeWeights(numNeurons, inputSize)
        if initialb:
            self.b = initialb
        else:
            self.b = initializeWeights(numNeurons, 1)