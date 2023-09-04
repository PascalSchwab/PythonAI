from activeFunc import ActiveFunc
from math import e


class Neuron:
    def __init__(self) -> None:
        self.input = 0
        self.output = 0
        self.weights = []
        self.derivedWeights = [0, 0]
        self.error = 0
        self.derivedError = 0

    def calcOutput(self, activeFunc: ActiveFunc) -> None:
        if activeFunc == ActiveFunc.LINEAR:
            self.output = self.input
        elif activeFunc == ActiveFunc.SIGMOID:
            self.output = 1 / (1 + e ** (-self.input))
        elif activeFunc == ActiveFunc.RELU:
            self.output = self.input
