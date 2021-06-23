import os
import random

class TrainingSetLoader(object):
    """
        Class to handle the training set by batches
    """
    def __init__(self, trainingSetFolder):
        self.__createPaths(trainingSetFolder)

        self.__batchesFiles = os.listdir(self.__pickleTrainingSetFolder)

        random.shuffle(self.__batchesFiles)

        self.__numberBatches = len(self.__batchesFiles)
        self.__currentBatchCounter = 0

    def __createPaths(self, trainingSetFolder):
        """
            Method to create relevant paths
        """
        dirpath = os.path.dirname(__file__)
        self.__path = os.path.split(dirpath)[0]
        self.__picklePath = os.path.join(self.__path, "pickleFolder")
        self.__pickleTrainingSetFolder = os.path.join(self.__picklePath, trainingSetFolder)

        self.__createFolder(self.__picklePath)
        self.__createFolder(self.__pickleTrainingSetFolder)

    def getNumberBatches(self):
        """
            Method to obtain the number of batches
        """
        return self.__numberBatches

    def getCurrentNumberBatch(self):
        """
            Method to obtain the number of batches
        """
        return self.__currentBatchCounter

    def getCurrentBatch(self):
        """
            Method to obtain current batch and increase the counter
        """
        currentBatch = os.path.join(
                                    self.__pickleTrainingSetFolder,
                                    self.__batchesFiles[self.__currentBatchCounter]
                                    )
        self.__currentBatchCounter += 1

        if self.__currentBatchCounter == self.__numberBatches:
            self.__currentBatchCounter = 0
            random.shuffle(self.__batchesFiles)

        return currentBatch