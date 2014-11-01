from abc import ABCMeta, abstractmethod


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
    def __init__(self, trainer, datasetForTraining, datasetForTest, epochs):
        self.trainer = trainer
        self.datasetForTraining = datasetForTraining
        self.datasetForTest = datasetForTest
        self.epochs = epochs

        self.collectedErrors = []

    def execute(self):
        self.trainer.trainOnDataset(self.datasetForTraining, self.epochs)
        averageError = self.trainer.testOnData(self.datasetForTest)
        self.collectedErrors.append(averageError)

        return averageError

    def getMinError(self):
        return min(self.collectedErrors)

    def getMaxError(self):
        return max(self.collectedErrors)

    def getAverageError(self):
        return sum(self.collectedErrors) / len(self.collectedErrors)