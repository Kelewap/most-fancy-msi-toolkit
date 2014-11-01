from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork

from my_stuff import YourCoolMsiDataProvider, thingsYouDoWithTheNeuralNetwork


myDataProvider = YourCoolMsiDataProvider()
entryDimension = myDataProvider.getInputDimension()
resultDimension = myDataProvider.getOutputDimension()
trainingData = myDataProvider.getTrainingData()
testData = myDataProvider.getTestData()
hiddenLayerDimension = myDataProvider.getNetworkHiddenLayerDimension()

datasetForTraining = SupervisedDataSet(entryDimension, resultDimension)
for entry, expectedResult in trainingData:
    datasetForTraining.addSample(entry, expectedResult)

datasetForTest = SupervisedDataSet(entryDimension, resultDimension)
for entry, expectedResult in testData:
    datasetForTest.addSample(entry, expectedResult)

network = buildNetwork(entryDimension, hiddenLayerDimension, resultDimension, recurrent=True)

thingsYouDoWithTheNeuralNetwork(network, datasetForTraining, datasetForTest)
