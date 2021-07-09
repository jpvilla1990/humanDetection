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
        self.__maxSize = 512
        self.__threshold = 0.2

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
        image = Image.open(os.path.join(self.__predictionsPath, imagePath))
        imageTorch = self.__toTensor(image)

        return imageTorch

    def __loadImageAbsPath(self, imagePath):
        """
            Load an Image from disk to torch
            returns a normalized image
        """
        image = Image.open(imagePath)
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
        imageTorch = self.__loadImageAbsPath(imageFile)
        if imageTorch.shape[0] != 3 and len(imageTorch.shape) == 3:
            newImageTorch = torch.ones([3, imageTorch.shape[1], imageTorch.shape[2]])
            newImageTorch[0] = imageTorch
            newImageTorch[1] = imageTorch
            newImageTorch[2] = imageTorch
            imageTorch = newImageTorch

        elif len(imageTorch.shape) == 2:
            newImageTorch = torch.ones([3, imageTorch.shape[0], imageTorch.shape[1]])
            newImageTorch[0] = imageTorch
            newImageTorch[1] = imageTorch
            newImageTorch[2] = imageTorch
            imageTorch = newImageTorch

        if imageTorch.shape[1] > imageTorch.shape[2]:
            maxSize = [self.__maxSize, int(self.__maxSize * (imageTorch.shape[2] / imageTorch.shape[1]))]
        else:
            maxSize = [int(self.__maxSize * (imageTorch.shape[1] / imageTorch.shape[2])), self.__maxSize]

        if imageTorch.shape[1] > maxSize[0]:
            imageTorch = transforms.functional.resize(imageTorch, (maxSize[0], imageTorch.shape[2]))
        if imageTorch.shape[2] > maxSize[1]:
            imageTorch = transforms.functional.resize(imageTorch, (imageTorch.shape[1], maxSize[1]))

        imagesCropped, dimensions = self.__cropImage(imageTorch)

        parameters = self.__loadPickle(os.path.join(self.__pickleModelsPath, modelFile))

        img = imagesCropped.permute(0,2,3,1)

        croppedPrediction = model.forward(img, parameters)

        croppedPrediction = torch.sigmoid(croppedPrediction)

        ann_hat = torch.flatten(croppedPrediction, start_dim=1)
        dimX = self.__imageSize[1]
        dimY = int(ann_hat.shape[1] / dimX)

        ann_hat = ann_hat.reshape([ann_hat.shape[0], 1, dimX, dimY])
        ann_hat = torch.nn.functional.interpolate(ann_hat, (self.__imageSize[0], self.__imageSize[1]))

        return ann_hat, dimensions

    def reconstructImage(self, croppedImage, dimensions):
        """
            Reconstruct a cropped Image
        """
        desiredHeight = dimensions[0]
        desiredWidth = dimensions[1]

        divisionsHeight = int(math.floor(desiredHeight / self.__imageSize[0]))
        divisionsWidth = int(math.floor(desiredWidth / self.__imageSize[1]))

        targetImage = torch.ones([1, desiredHeight, desiredWidth])

        cropsHeight = divisionsHeight + 1
        cropsWidth = divisionsWidth + 1
        numberCrops = (cropsHeight) * (cropsWidth)

        if numberCrops != croppedImage.shape[0]:
            print("ERROR: number of detected crops does not coincide with number of expected crops")
            exit()

        if cropsHeight == 1 and cropsWidth == 1:
            targetImage = transforms.functional.resize(croppedImage[0], [desiredHeight, desiredWidth])

        elif cropsHeight == 1 and cropsWidth > 1:
            for i in range(cropsWidth - 1):
                xInit = i * self.__imageSize[1]
                xEnd = (i + 1) * self.__imageSize[1]
                croppedResized = transforms.functional.resize(croppedImage[i], [desiredHeight, self.__imageSize[1]])
                targetImage[0, :, xInit: xEnd] = croppedResized

            croppedResized = transforms.functional.resize(croppedImage[cropsWidth], [desiredHeight, self.__imageSize[1]])
            targetImage[0, :, desiredWidth - self.__imageSize[1]: desiredWidth] = croppedResized

        elif cropsHeight > 1 and cropsWidth == 1:
            for i in range(cropsHeight - 1):
                yInit = i * self.__imageSize[0]
                yEnd = (i + 1) * self.__imageSize[0]
                croppedResized = transforms.functional.resize(croppedImage[i], [self.__imageSize[0], desiredWidth])
                targetImage[0, yInit: yEnd, :] = croppedResized

            croppedResized = transforms.functional.resize(croppedImage[cropsHeight], [self.__imageSize[0], desiredWidth])
            targetImage[0, desiredHeight - self.__imageSize[0]:desiredHeight, :] = croppedResized

        elif cropsHeight > 1 and cropsWidth > 1:
            cropIndex = 0
            for i in range(cropsHeight):
                for j in range(cropsWidth):
                    yInit = i * self.__imageSize[0]
                    yEnd = (i + 1) * self.__imageSize[0]
                    xInit = j * self.__imageSize[1]
                    xEnd = (j + 1) * self.__imageSize[1]

                    if i == cropsHeight - 1 and j != cropsWidth - 1:
                        targetImage[0, desiredHeight - self.__imageSize[0]: desiredHeight, xInit: xEnd] = croppedImage[cropIndex]
                    elif j == cropsWidth - 1 and i != cropsHeight - 1:
                        targetImage[0, yInit: yEnd, desiredWidth - self.__imageSize[1]: desiredWidth] = croppedImage[cropIndex]
                    elif j == cropsWidth - 1 and i == cropsHeight - 1:
                        targetImage[0, desiredHeight - self.__imageSize[0]: desiredHeight, desiredWidth - self.__imageSize[1]: desiredWidth] = croppedImage[cropIndex]
                    else:
                        targetImage[0, yInit: yEnd, xInit: xEnd] = croppedImage[cropIndex]

                    cropIndex += 1

        return targetImage

    def postProcessImage(self, prediction):
        """
            Method to convert torch to PIL and do postprocessing on the image
        """
        prediction[prediction > self.__threshold] = 1.0
        prediction[prediction <= self.__threshold] = 0.0
        image = self.__toPIL(prediction)
        return image
