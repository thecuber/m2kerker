import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:

    def __init__(self, layers):
        self.weights = [np.random.randn(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        self.bias = [np.random.randn(layers[i]) for i in range(1, len(layers))]
        self.size = len(layers)
        self.sizes = layers

    def sigmoid(self, z):
        return np.vectorize(lambda x : 1/(1 + np.exp(-x)))(z)

    def sigmoidp(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))



    def passData(self, input):
        self.y = []
        self.z = [input]
        for i in range(self.size - 1):
            self.y.append(np.dot(self.z[-1], self.weights[i]) + self.bias[i])
            self.z.append(self.sigmoid(self.y[-1]))

    def expandArray(self, arr, axis, count):
        z = []
        #axis == 0:expand the array horizontally ( for a column array)
        #axis == 1: expand the array vertically ( for a line array)
        if axis == 0:
            for i in range(len(arr)):
                z.append(np.full((count), arr[i][0]))
        else:
            #print("#♠2", arr, type(arr))
            for i in range(count):
                z.append(arr)
        return np.array(z)

    def trainNetwork(self, data, eta, n):
        for i in range(n):
            input, target = data[np.random.randint(len(data))]
            self.passData(input)
            gamma = [0. for i in range(self.size - 1)]
            #print("#1", self.weights[0])
            gamma[-1] = self.z[-1] - target
            for j in range(self.size - 2, -1, -1):
                if(j != 0):#backprop the error
                    gamma[j - 1] = np.dot(self.weights[j], np.transpose(self.sigmoidp(self.y[j]) * gamma[j]))
                #print("#4", gamma[-1], self.y[0])
                #print("#3", self.expandArray(gamma[j] * self.sigmoidp(self.y[j]), 1, self.sizes[j]), "/" *10, "/"*10, self.expandArray(self.z[j].reshape(-1, 1), 0, self.sizes[j + 1]))
                self.weights[j] = self.weights[j] - eta * self.expandArray(gamma[j] * self.sigmoidp(self.y[j]), 1, self.sizes[j]) * self.expandArray(self.z[j].reshape(-1, 1), 0, self.sizes[j + 1])
                self.bias[j] =  self.bias[j] - eta * gamma[j] * self.sigmoidp(self.y[j])
        print("RESULTS")
        for i in range(10):
            input, target = data[np.random.randint(len(data))]
            self.passData(input)
            print("Expected ->", target, ", output ->", self.z[-1])


r = 4*np.random.rand(2000, 2) - 2
data = [[i, np.array([(i[0]**2 + i[1]**2)>=1])] for i in r]
n = NeuralNetwork((2, 4, 1))
n.trainNetwork(data, 0.0001, 100000)

