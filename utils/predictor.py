import os
import torch
import pickle
import math
import random
from torchvision import transforms
from PIL import Image

from utils.model import SwinBlocks

class Predictor(object):
    """
        Class to perform predictions over a trained model
    """
    def __init__(self, imageWidth=256, imageHeight=256):
        self.__imageSize = [imageWidth, imageHeight]
        self.__toPIL = transforms.ToPILImage()
        self.__toTensor = transforms.ToTensor()
        self.__createPaths()
        self.__lastTime = None

    def __createFolder(self, folder):
        """
            Method to create a folder
        """
        try:
            os.mkdir(folder)
        except:
            pass

    def __createPaths(self):
        """
            Method to create relevant paths
        """
        dirpath = os.path.dirname(__file__)
        self.__path = os.path.split(dirpath)[0]
        self.__picklePath = os.path.join(self.__path, "pickleFolder")
        self.__pickleModelsPath = os.path.join(self.__picklePath, "pickleModels")
        self.__predictionsPath = os.path.join(self.__path, "predictionsPath")

        self.__createFolder(self.__picklePath)
        self.__createFolder(self.__pickleModelsPath)
        self.__createFolder(self.__predictionsPath)
    
    def __loadPickle(self, file):
        """
            Method to load a pickel file
        """
        with open(file, 'rb') as f:
            return pickle.load(f)

    def __loadImage(self, imagePath):
        """
            Load an Image from disk to torch
            returns a normalized image
        """
        image = Image.open(os.path.join(self.__predictionsPath), imagePath)
        imageTorch = self.__toTensor(image)

        return imageTorch

    def __cropImage(self, imageTorch):
        """
            Method to crop image in pieces with the desired size
        """
        totalHeight = imageTorch.shape[1]
        totalWidth = imageTorch.shape[2]

        divisionsWidth = int(math.floor(totalWidth / self.__imageSize[0]))
        divisionsHeight = int(math.floor(totalHeight / self.__imageSize[1]))

        if divisionsWidth == 0:
            imageTorch = transforms.functional.resize(imageTorch, [totalHeight, self.__imageSize[0]])
            divisionsWidth = 1
            totalHeight = imageTorch.shape[1]
            totalWidth = imageTorch.shape[2]

        if divisionsHeight == 0:
            imageTorch = transforms.functional.resize(imageTorch, [self.__imageSize[1], totalWidth])
            divisionsHeight = 1
            totalHeight = imageTorch.shape[1]
            totalWidth = imageTorch.shape[2]

        numberCrops = (divisionsWidth + 1) * (divisionsHeight + 1)

        cropImages = torch.ones([numberCrops, imageTorch.shape[0], self.__imageSize[0], self.__imageSize[1]])

        tops = [i * self.__imageSize[1] for i in range(divisionsHeight)]
        tops.append(totalHeight - self.__imageSize[1])

        lefts = [i * self.__imageSize[0] for i in range(divisionsWidth)]
        lefts.append(totalWidth - self.__imageSize[0])

        cropsIndex = 0

        for top in tops:
            for left in lefts:
                cropped = transforms.functional.crop(imageTorch, top, left, self.__imageSize[1], self.__imageSize[0])
                cropImages[cropsIndex] = cropped

                cropsIndex += 1

        return cropImages, [totalHeight, totalWidth]

    def predictor(self, imageFile, modelFile="supervisedLearning.pickle", model=None):
        """
            Method to predict mask of an image
            return predictions, [height, width]
        """
        imageTorch = self.__loadImage(imageFile)
        if imageTorch.shape[0] != 3:
            newImageTorch = torch.ones([3, imageTorch.shape[1], imageTorch.shape[2]])
            newImageTorch[0] = imageTorch
            newImageTorch[1] = imageTorch
            newImageTorch[2] = imageTorch
            imageTorch = newImageTorch

        imagesCropped, dimensions = self.__cropImage(imageTorch)

        parameters = self.__loadPickle(os.path.join(self.__pickleModelsPath, modelFile))

        img = img.permute(0,2,3,1)

        croppedPrediction = model.forward(img, parameters)

        croppedPrediction = torch.sigmoid(croppedPrediction)

        return croppedPrediction, dimensions

    def reconstructImage(self, croppedImage, dimensions):
        """
            Reconstruct a cropped Image
        """
        desiredHeight = dimensions[0]
        desiredWidth = dimensions[1]