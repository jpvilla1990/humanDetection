from utils.dataloader import Dataloader
from utils.dataPreprocessing import DataPreprocessing

class Main(object):
    def __init__(self, category="person"):
        self.__category="person"

    def __runDataloader(self):
        """
            Method to run dataloader
        """
        dataloader = Dataloader(self.__category)
        dataloader.downloadCocoDataset()

    def run(self):
        """
            Method to run all processes
        """
        self.__runDataloader()
