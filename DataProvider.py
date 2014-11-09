from numpy import arange
from pybrain.supervised import BackpropTrainer
import sys
from BatchTrainingExecutor import AbstractMsiDataProvider, BatchTrainingExecutor


class YourCoolMsiDataProvider(AbstractMsiDataProvider):
    def __init__(self):
        rates = []

        with open("rates.csv", "rt") as ratesFile:
            for line in ratesFile:
                dateString, rateString = line.strip().split(",")[:2]
                rate = float(rateString)
                rates.append(rate)
        rates = list(reversed(rates))
        rates = rates[:49]
        print rates
        # crazy fking important - normalization
        self.maxRates = max(rates)
        rates = map(lambda el: el / max(rates), rates)
        self.allLearningDataset = rates

        self.previousRatesCountUsedForPrediction = 10
        self.fineMachineLearningDataset = []
        for i in xrange(self.previousRatesCountUsedForPrediction, len(rates)):
            thisEntry = rates[i - self.previousRatesCountUsedForPrediction: i]
            expectedResultForThisEntry = rates[i]

            self.fineMachineLearningDataset.append((thisEntry, [expectedResultForThisEntry, ]))

    def getAllLearningDataset(self):
        return self.allLearningDataset

    def getInputDimension(self):
        return self.previousRatesCountUsedForPrediction

    def getOutputDimension(self):
        return 1

    def getTrainingData(self):
        return self.fineMachineLearningDataset[:-10]

    def getTestData(self):
        return self.fineMachineLearningDataset[-10:]

    def getNetworkHiddenLayerDimension(self):
        return 4

    def initialMaxRates(self):
        return self.maxRates


def thingsYouDoWithTheNeuralNetwork(networkFactoryMethod, datasetForTraining, datasetForTest, allTraingDataset,
                                    dataProvider):
    # write your fancy scientific experiment below

    # epochsToTest = 10
    for epochsToTest in range(20, 21, 20):
        executor = BatchTrainingExecutor(networkFactoryMethod, datasetForTraining, datasetForTest, allTraingDataset,
                                         dataProvider, epochs=epochsToTest, learningrate=0.01, momentum=0.08)
        print >> sys.stderr, "executing for epochs", epochsToTest
        for i in xrange(6):
            print >> sys.stderr, "executing", i + 1, "iteration"
            # executor.execute()
            executor.predictNextElements(20)

            # thingsToPrint = {
            #     "variable": epochsToTest,
            #     "minError": executor.getMinError(),
            #     "maxError": executor.getMaxError(),
            #     "averageError": executor.getAverageError()
            # }
            # print "{variable},{minError},{maxError},{averageError}".format(**thingsToPrint)
        executor.getMeanResults()
        executor.saveResultsToFiles()

