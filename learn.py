from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork

from DataProvider import YourCoolMsiDataProvider, thingsYouDoWithTheNeuralNetwork


myDataProvider = YourCoolMsiDataProvider()
entryDimension = myDataProvider.getInputDimension()
resultDimension = myDataProvider.getOutputDimension()
trainingData = myDataProvider.getTrainingData()
testData = myDataProvider.getTestData()
hiddenLayerDimension = myDataProvider.getNetworkHiddenLayerDimension()
allTraingDataset = myDataProvider.getAllLearningDataset()

datasetForTraining = SupervisedDataSet(entryDimension, resultDimension)
for entry, expectedResult in trainingData:
    datasetForTraining.addSample(entry, expectedResult)

datasetForTest = SupervisedDataSet(entryDimension, resultDimension)
for entry, expectedResult in testData:
    datasetForTest.addSample(entry, expectedResult)


def networkFactoryMethod():
    return buildNetwork(entryDimension, hiddenLayerDimension, resultDimension, recurrent=True, bias=True)


thingsYouDoWithTheNeuralNetwork(networkFactoryMethod, datasetForTraining, datasetForTest, allTraingDataset,
                                myDataProvider)

