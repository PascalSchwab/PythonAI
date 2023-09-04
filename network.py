from layer import Layer
from dataModel import DataModel


class Network:
    def __init__(self, layers) -> None:
        self.layers = layers
        self.learnRate = 0.01

        self.initNetwork()

    def train(self, models, periods) -> None:
        for _ in range(periods):
            for currentModel in models:
                self.setData(currentModel)
                self.forwardPropagation()
                self.backwardPropagation(currentModel)
                self.updateNetwork()

    def test(self, model: DataModel) -> None:
        pass

    def initNetwork(self) -> None:
        # Init weights
        self.layers[0].neurons[0].weights.append(0.1)
        self.layers[0].neurons[0].weights.append(0.2)
        self.layers[0].neurons[1].weights.append(0.3)
        self.layers[0].neurons[1].weights.append(0.4)
        self.layers[0].neurons[2].weights.append(0.5)
        self.layers[0].neurons[2].weights.append(0.6)

        self.layers[1].neurons[0].weights.append(0.7)
        self.layers[1].neurons[0].weights.append(0.8)
        self.layers[1].neurons[1].weights.append(0.9)
        self.layers[1].neurons[1].weights.append(0.1)

    def setData(self, model: DataModel) -> None:
        for i in range(len(self.layers)):
            self.layers[i].derivedBias = 0
            for j in range(self.layers[i].neuronCount):
                if i == 0:
                    self.layers[i].neurons[j].input = model.input[j]
                else:
                    self.layers[i].neurons[j].input = 0
                self.layers[i].neurons[j].output = 0
                self.layers[i].neurons[j].error = 0
                self.layers[i].neurons[j].derivedError = 0
                self.layers[i].neurons[j].derivedWeights[0] = 0
                self.layers[i].neurons[j].derivedWeights[1] = 0

    def forwardPropagation(self) -> None:
        for i in range(len(self.layers)):
            for j in range(self.layers[i].neuronCount):
                if i != 0:
                    for k in range(self.layers[i - 1].neuronCount):
                        self.layers[i].neurons[j].input += self.layers[i - 1].neurons[k].weights[j] * self.layers[i - 1].neurons[k].output
                    self.layers[i].neurons[j].input += self.layers[i].bias
                self.layers[i].neurons[j].calcOutput(self.layers[i].activeFunc)

    def backwardPropagation(self, model: DataModel) -> None:
        # Calculate DerivedError
        for i in reversed(range(len(self.layers))):
            for j in range(self.layers[i].neuronCount):
                self.layers[i].neurons[j].derivedError = self.layers[i].neurons[j].output * (1 - self.layers[i].neurons[j].output)
        # Calculate Error
        for i in reversed(range(len(self.layers))):
            for j in range(self.layers[i].neuronCount):
                if i == len(self.layers) - 1:
                    self.layers[i].neurons[j].error = self.layers[i].neurons[j].output - model.output[j]
                elif i != 0:
                    for k in range(self.layers[i + 1].neuronCount):
                        self.layers[i].neurons[j].error += self.layers[i + 1].neurons[k].error * self.layers[i + 1].neurons[k].derivedError * self.layers[i].neurons[j].weights[k]
        # Calculate Weights
        for i in reversed(range(len(self.layers))):
            for j in range(self.layers[i].neuronCount):
                if i != len(self.layers) - 1:
                    for k in range(self.layers[i+1].neuronCount):
                        self.layers[i].neurons[j].derivedWeights[k] = self.layers[i+1].neurons[k].error*self.layers[i+1].neurons[k].derivedError*self.layers[i].neurons[j].output

    def updateNetwork(self) -> None:
        for i in range(len(self.layers)):
            for j in range(self.layers[i].neuronCount):
                for k in range(len(self.layers[i].neurons[j].weights)):
                    self.layers[i].neurons[j].weights[k] = self.layers[i].neurons[j].weights[k] - self.learnRate*self.layers[i].neurons[j].derivedWeights[k]