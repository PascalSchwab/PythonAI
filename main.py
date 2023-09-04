from network import Network
from layer import Layer
from activeFunc import ActiveFunc
from dataModel import DataModel

trainData = [DataModel([1, 4, 5], [0.1, 0.05])]
testData = []

layers = [
    Layer(3, 1.0, ActiveFunc.LINEAR),
    Layer(2, 0.5, ActiveFunc.SIGMOID),
    Layer(2, 0.5, ActiveFunc.SIGMOID),
]

network = Network(layers)

network.train(trainData, 1)
