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
        # print rates
        # crazy fking important - normalization
        self.maxRates = max(rates)
        rates = map(lambda el: el / max(rates), rates)
        self.allLearningDataset = rates

        self.previousRatesCountUsedForPrediction = 20
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


