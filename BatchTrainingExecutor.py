from abc import ABCMeta, abstractmethod
from pybrain.supervised import BackpropTrainer


class AbstractMsiDataProvider(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getNetworkHiddenLayerDimension(self):
        pass

    @abstractmethod
    def getAllLearningDataset(self):
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
    def __init__(self, networkFactoryMethod, datasetForTraining, datasetForTest, allTrainingDataset,
                 dataProvider, epochs, learningrate, momentum):
        self.networkFactoryMethod = networkFactoryMethod
        self.datasetForTraining = datasetForTraining
        self.datasetForTest = datasetForTest
        self.epochs = epochs
        self.learningrate = learningrate
        self.momentum = momentum
        self.allTrainingDataset = allTrainingDataset

        self.dataProvider = dataProvider

        self.collectedErrors = []
        self.collectedResults = []

    def execute(self):
        network = self.networkFactoryMethod()
        trainer = BackpropTrainer(network, learningrate=self.learningrate, momentum=self.momentum)
        trainer.trainOnDataset(self.datasetForTraining, self.epochs)
        averageError = trainer.testOnData(self.datasetForTest)
        self.collectedErrors.append(averageError)

        results = []

        for x in self.allTrainingDataset[0:self.dataProvider.getInputDimension()]:
            print x
            results.append(x)

        for i in range(0, len(self.allTrainingDataset) - self.dataProvider.getInputDimension() - 1, 1):
            print 'current index of element predicted: ' + str(i + self.dataProvider.getInputDimension() + 1)
            print "using elements for prediction:"
            print self.allTrainingDataset[i:i + self.dataProvider.getInputDimension()]
            # print results[-self.dataProvider.getInputDimension():]
            x = network.activate(self.allTrainingDataset[i:i + self.dataProvider.getInputDimension()])
            print 'should be : ' + str(self.allTrainingDataset[i + self.dataProvider.getInputDimension() + 1])
            print 'got : ' + str(x)
            results.append(str(x[0]))

        self.results = results
        return averageError

    def predictNextElements(self, elementsToPredict):

        network = self.networkFactoryMethod()
        trainer = BackpropTrainer(network, learningrate=self.learningrate, momentum=self.momentum, )
        trainer.trainUntilConvergence(self.datasetForTraining, self.epochs)

        results = []

        for x in self.allTrainingDataset:
            print x
            results.append(x)

        for i in range(1, elementsToPredict, 1):
            print "using elements for prediction:"
            print results
            print results[-self.dataProvider.getInputDimension():]
            x = network.activate(results[-self.dataProvider.getInputDimension()])
            print "found new element: " + str(x)
            if x > 1:
                x -= 1

            results.append(x[0])

        self.collectedResults.append(results)
        print 'collected results'
        print self.collectedResults

    def getMinError(self):
        return min(self.collectedErrors)

    def getMaxError(self):
        return max(self.collectedErrors)

    def getAverageError(self):
        return sum(self.collectedErrors) / len(self.collectedErrors)

    def getMeanResults(self):
        results = []
        print 'len of single collected result list'
        print len(self.collectedResults[0])
        for i in range(0, len(self.collectedResults[0]), 1):
            p = []
            for row in self.collectedResults:
                p.append(row[i])

            print sum(p)
            print len(p)
            mean = sum(p) / len(p)
            results.append(mean)

        print ' Got mean values!!!!!!!!!!!!!!!!!!1'
        print results
        return results

    def savePredictedResultsToFiles(self):
        f = open('C:\\home\\aaaaStudia\\Semestr_VII\\MSI\\Lab2\\zad\\predictions_' + str(self.epochs) + '.data', 'w')
        for line in self.collectedResults[0]:
            # for line in self.getMeanResults():
            print line
            f.write(str(line) + "\n")

    def saveToFiles(self):
        f = open('C:\\home\\aaaaStudia\\Semestr_VII\\MSI\\Lab2\\zad\\my_' + str(self.epochs) + '.data', 'w')
        for line in self.results:
            # for line in self.getMeanResults():
            print line
            f.write(str(line) + "\n")

