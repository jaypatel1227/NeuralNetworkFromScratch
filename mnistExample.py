import numpy as np
import pandas as pd
import sequentialNetwork
import activationFunctions

data = pd.read_csv('mnist.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into test and training sets

data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255. 

data_train = data[1000:].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
n_train,m_train = X_train.shape

# I have tried both ReLU, tanh, and sigmoid as activation functions
# it appers that sigmoid will perform much better if the learning rate is very high
# tanh seems to perform much worse than the other two activation functions
# while ReLU does alright even if the learning rate is low though it seems to match sigmoid when the learning rate is high
network = sequentialNetwork.Network(n_train, 10, learningRate=0.8)
network.addLayer(sequentialNetwork.Layer(10,784,activationFunctions.ReLU, activationFunctions.derivReLU))
# network.addLayer(sequentialNetwork.Layer(10,10,activationFunctions.ReLU, activationFunctions.derivReLU))
network.addLayer(sequentialNetwork.Layer(10,10,activationFunctions.softMax))

# print(network.layers[0].W.shape)
# print(sequentialNetwork.oneHot(Y_train).shape)

network.fit(X_train, Y_train, 1000)

_ , As = network.forward(X_test)
preds = np.argmax(As[-1],0)
test_acc = np.sum(preds == Y_test) / Y_test.size
print("Testing Accuracy: ", test_acc) # this is about 91% for tests I have run with ReLU