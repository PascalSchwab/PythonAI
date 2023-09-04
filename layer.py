from neuron import Neuron
from activeFunc import ActiveFunc

class Layer:
    def __init__(self, neuronCount, bias, activeFunc: ActiveFunc) -> None:
        self.neuronCount = neuronCount
        self.bias = bias
        self.activeFunc = activeFunc
        self.derivedBias = 0

        self.initNeurons()

    def initNeurons(self) -> None:
        self.neurons = []
        for _ in range(self.neuronCount):
            self.neurons.append(Neuron())