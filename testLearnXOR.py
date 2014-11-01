from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

ENTRY_DIMENSION = 2
RESULT_DIMENSION = 1

trainingData = (
    ((0.0, 0.0), 0.0),
    ((1.0, 0.0), 1.0),
    ((1.0, 1.0), 0.0),
    ((0.0, 1.0), 1.0),
    ((0.1, 0.9), 1.0),
    ((0.2, 0.9), 1.0),
    ((0.2, 0.3), 0.0),
    ((0.4, 0.0), 0.0),
    ((0.4, 0.2), 0.0),
    ((0.7, 0.8), 0.0),
    ((0.1, 0.8), 1.0),
    ((0.3, 1.0), 1.0),
    ((1.0, 0.6), 0.0),
    ((0.7, 0.6), 0.0),
    ((0.7, 0.1), 1.0),
)

testData = (
    ((0.8, 0.0), 1.0),
    ((0.9, 0.7), 0.0),
    ((0.1, 0.1), 0.0),
    ((0.2, 0.8), 1.0),
    ((0.6, 0.6), 0.0),
    ((0.6, 1.0), 0.0),
    ((1.0, 0.3), 1.0),
    ((0.1, 0.1), 0.0),
)

datasetForTraining = SupervisedDataSet(ENTRY_DIMENSION, RESULT_DIMENSION)
for entry, expectedResult in trainingData:
    datasetForTraining.addSample(entry, [expectedResult])

datasetForTest = SupervisedDataSet(ENTRY_DIMENSION, RESULT_DIMENSION)
for entry, expectedResult in testData:
    datasetForTest.addSample(entry, [expectedResult])

HIDDEN_LAYER_DIMENSION = 4
network = buildNetwork(ENTRY_DIMENSION, HIDDEN_LAYER_DIMENSION, RESULT_DIMENSION, recurrent=True)
trainer = BackpropTrainer(network, learningrate=0.01, momentum=0.99, verbose=True)
trainer.trainOnDataset(datasetForTraining, 1)
trainer.testOnData(datasetForTest, verbose=True)