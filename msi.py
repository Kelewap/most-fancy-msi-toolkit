from abc import ABCMeta, abstractmethod
from pybrain.supervised import BackpropTrainer


class AbstractMsiDataProvider(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getNetworkHiddenLayerDimension(self):
        pass

    @abstractmethod
    def getInputDimension(self):
        pass

    @abstractmethod
    def getOutputDimension(self):
        pass

    @abstractmethod
    def getTrainingData(self):
        pass

    @abstractmethod
    def getTestData(self):
        pass


class BatchTrainingExecutor(object):
    def __init__(self, networkFactoryMethod, datasetForTraining, datasetForTest, epochs, learningrate, momentum):
        self.networkFactoryMethod = networkFactoryMethod
        self.datasetForTraining = datasetForTraining
        self.datasetForTest = datasetForTest
        self.epochs = epochs
        self.learningrate = learningrate
        self.momentum = momentum

        self.collectedErrors = []

    def execute(self):
        network = self.networkFactoryMethod()
        trainer = BackpropTrainer(network, learningrate = self.learningrate, momentum = self.momentum)
        trainer.trainOnDataset(self.datasetForTraining, self.epochs)
        averageError = trainer.testOnData(self.datasetForTest)
        self.collectedErrors.append(averageError)

        return averageError

    def getMinError(self):
        return min(self.collectedErrors)

    def getMaxError(self):
        return max(self.collectedErrors)

    def getAverageError(self):
        return sum(self.collectedErrors) / len(self.collectedErrors)