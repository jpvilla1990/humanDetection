from utils.dataloader import Dataloader
from utils.dataPreprocessing import DataPreprocessing
from utils.model import SwinBlocks
from utils.train import Train

class Main(object):
    def __init__(self, category="person"):
        self.__category="person"

    def __runDataloader(self):
        """
            Method to run dataloader
        """
        dataloader = Dataloader(self.__category)
        dataloader.downloadCocoDataset()

    def runTrain(self, lr=0.001):
        """
            Method to run all processes
        """
        self.__runDataloader()
        model = SwinBlocks()
        train = Train()
        train.supervisedLearningTrain(model=model, lr=lr)

    def getTrainStatus(self, lines):
        """
            Method to obtain train status
        """
        train = Train()
        return train.returnLog(lines=lines)

    def __rescaleVector(self, vector, scale):
        """
            Rescale a vector to make it shorter
        """
        lenNewVector = int(len(vector) / scale)
        newVector = []
        for i in range(lenNewVector):
            init = i * scale
            final = (i + 1) * scale
            bufferVector = vector[init: final]
            avg = sum(bufferVector)/len(bufferVector)
            newVector.append(avg)

    def getLoss(self):
        """
            Method to obtain an array of len=100 with the loss into log scale
        """
        train = Train()
        loss = train.returnLoss()
        scale = len(loss) / 100
        lossScaled = self.__rescaleVector(loss, scale)
        return lossScaled
