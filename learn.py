from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
import sys
from BatchTrainingExecutor import BatchTrainingExecutor

from DataProvider import YourCoolMsiDataProvider


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





def thingsYouDoWithTheNeuralNetwork(networkFactoryMethod, datasetForTraining, datasetForTest, allTraingDataset,
                                    dataProvider):
    # write your fancy scientific experiment below

    # epochsToTest = 10
    for epochsToTest in range(20, 21, 20):
        executor = BatchTrainingExecutor(networkFactoryMethod, datasetForTraining, datasetForTest, allTraingDataset,
                                         dataProvider, epochs=epochsToTest, learningrate=0.004, momentum=0.08)
        print >> sys.stderr, "executing for epochs", epochsToTest
        for i in xrange(6):
            print >> sys.stderr, "executing", i + 1, "iteration"
            # executor.execute()
            executor.predictNextElements(20)
        #
        # thingsToPrint = {
        #     "variable": epochsToTest,
        #     "minError": executor.getMinError(),
        #     "maxError": executor.getMaxError(),
        #     "averageError": executor.getAverageError()
        # }
        # print "{variable},{minError},{maxError},{averageError}".format(**thingsToPrint)
        # executor.saveToFiles()
        executor.getMeanResults()
        executor.savePredictedResultsToFiles()



thingsYouDoWithTheNeuralNetwork(networkFactoryMethod, datasetForTraining, datasetForTest, allTraingDataset,
                                myDataProvider)
