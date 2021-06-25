import os
import sys
import math
import random
import pickle
from PIL import Image
import torch
from torchvision import transforms

class DataPreprocessing(object):
    def __init__(self, category="person", batchSize=5, imageWidth=256, imageHeight=256):
        self.__batchSize = batchSize
        self.__imageSize = [imageWidth, imageHeight]
        self.__createPaths(category)
        self.__toPIL = transforms.ToPILImage()
        self.__toTensor = transforms.ToTensor()

    def __createPaths(self, category):
        """
            Method to create paths
        """
        dirpath = os.path.dirname(__file__)
        self.__path = os.path.split(dirpath)[0]
        self.__datasetfolder = os.path.join(self.__path, "COCO")
        self.__train = os.path.join(self.__datasetfolder, "train")
        self.__val = os.path.join(self.__datasetfolder, "val")
        self.__test = os.path.join(self.__datasetfolder, "test")

        self.__annotations = os.path.join(self.__datasetfolder, "annotations")
        self.__annotations = os.path.join(self.__annotations, "annotations")
        self.__annotationsFileTrainJSON = os.path.join(self.__annotations, "instances_train2017.json")
        self.__annotationsFileValJSON = os.path.join(self.__annotations, "instances_val2017.json")

        self.__imagesTrainFolder = os.path.join(self.__train, "train2017")
        self.__imagesValFolder = os.path.join(self.__val, "val2017")
        
        self.__trainPersons = os.path.join(self.__train, category + "s")
        self.__trainPersonsImages = os.path.join(self.__trainPersons, "images")
        self.__trainPersonsAnn = os.path.join(self.__trainPersons, "annotations")

        self.__picklePath = os.path.join(self.__path, "pickleFolder")
        self.__trainPickleBatches = os.path.join(self.__picklePath, "pickleBatches")

        self.__createFolder(self.__datasetfolder)
        self.__createFolder(self.__train)
        self.__createFolder(self.__val)
        self.__createFolder(self.__test)
        self.__createFolder(self.__annotations)
        self.__createFolder(self.__trainPersons)
        self.__createFolder(self.__trainPersonsImages)
        self.__createFolder(self.__trainPersonsAnn)
        self.__createFolder(self.__picklePath)
        self.__createFolder(self.__trainPickleBatches)

    def __createFolder(self, folder):
        """
            Method to create a folder
        """
        try:
            os.mkdir(folder)
        except:
            pass

    def __loadImage(self, imagePath):
        """
            Load an Image from disk to torch
            returns a normalized image
        """
        image = Image.open(imagePath)
        imageTorch = self.__toTensor(image)

        return imageTorch

    def __cropImage(self, imageTorch, annotationTorch):
        """
            Method to crop image in pieces with the desired size
        """
        totalHeight = imageTorch.shape[1]
        totalWidth = imageTorch.shape[2]

        divisionsWidth = int(math.floor(totalWidth / self.__imageSize[0]))
        divisionsHeight = int(math.floor(totalHeight / self.__imageSize[1]))

        if divisionsWidth == 0:
            imageTorch = transforms.functional.resize(imageTorch, [totalHeight, self.__imageSize[0]])
            annotationTorch = transforms.functional.resize(annotationTorch, [totalHeight, self.__imageSize[0]])
            divisionsWidth = 1
            totalHeight = imageTorch.shape[1]
            totalWidth = imageTorch.shape[2]

        if divisionsHeight == 0:
            imageTorch = transforms.functional.resize(imageTorch, [self.__imageSize[1], totalWidth])
            annotationTorch = transforms.functional.resize(annotationTorch, [self.__imageSize[1], totalWidth])
            divisionsHeight = 1
            totalHeight = imageTorch.shape[1]
            totalWidth = imageTorch.shape[2]

        numberCrops = (divisionsWidth + 1) * (divisionsHeight + 1)

        missingSamples = self.__batchSize - (numberCrops % self.__batchSize)
        totalNumberCrops = numberCrops + missingSamples

        cropImages = torch.ones([totalNumberCrops, imageTorch.shape[0], self.__imageSize[0], self.__imageSize[1]])
        cropAnnotations = torch.ones([totalNumberCrops, annotationTorch.shape[0], self.__imageSize[0], self.__imageSize[1]])

        tops = [i * self.__imageSize[1] for i in range(divisionsHeight)]
        tops.append(totalHeight - self.__imageSize[1])

        lefts = [i * self.__imageSize[0] for i in range(divisionsWidth)]
        lefts.append(totalWidth - self.__imageSize[0])

        cropsIndex = 0

        for top in tops:
            for left in lefts:
                cropped = transforms.functional.crop(imageTorch, top, left, self.__imageSize[1], self.__imageSize[0])
                cropImages[cropsIndex] = cropped

                cropped = transforms.functional.crop(annotationTorch, top, left, self.__imageSize[1], self.__imageSize[0])
                cropAnnotations[cropsIndex] = cropped

                cropsIndex += 1

        for i in range(missingSamples):
            randomTop = random.randint(0, totalHeight - self.__imageSize[1])
            randomLeft = random.randint(0, totalWidth - self.__imageSize[0])

            cropped = transforms.functional.crop(imageTorch, randomTop, randomLeft, self.__imageSize[1], self.__imageSize[0])
            cropImages[cropsIndex] = cropped

            cropped = transforms.functional.crop(annotationTorch, randomTop, randomLeft, self.__imageSize[1], self.__imageSize[0])
            cropAnnotations[cropsIndex] = cropped

            cropsIndex += 1

        return cropImages, cropAnnotations

    def __splitInBatches(self, samples):
        """
            Method to split in batches
        """
        batches = []
        numberBatches = int(len(samples) / self.__batchSize)
        for i in range(numberBatches):
            batch = torch.ones([self.__batchSize, samples.shape[1], samples.shape[2], samples.shape[3]])
            batch = batch * samples[(i * self.__batchSize):((i + 1) * self.__batchSize)]
            batches.append(batch)

        return batches

    def __savePickleBatches(self, batchImages, batchAnnotations, imageName, batchName):
        """
            Store in a pickle file in an array in the form: [image, annotation, imageName]
        """
        with open(batchName, 'wb') as f:
            pickle.dump([batchImages, batchAnnotations, imageName], f)

    def __printProgress(self, index, total):
        """
            Show the progress in percentage
        """
        print("\rPROGRESS: {}%".format(round((index * 100) / total, 2)), end="", flush=True)

        if index == total - 1:
            print("\n")

    def createPickleBatches(self):
        """
            Method to create batches as pickle files
        """
        imageFiles = os.listdir(self.__trainPersonsImages)
        annotationFiles = os.listdir(self.__trainPersonsAnn)

        batchIndex = 0

        for index in range(len(imageFiles)):
            imageFile = os.path.join(self.__trainPersonsImages, imageFiles[index])
            annotationFile = os.path.join(self.__trainPersonsAnn, annotationFiles[index])

            imageTorch = self.__loadImage(imageFile)
            annotationTorch = self.__loadImage(annotationFile)

            imagesCropped, annotationsCropped = self.__cropImage(imageTorch, annotationTorch)

            batchesImages = self.__splitInBatches(imagesCropped)
            batchesAnnotations = self.__splitInBatches(annotationsCropped)

            for i in range(len(batchesImages)):
                pickleBatchFile = os.path.join(self.__trainPickleBatches, "batch_{}.pickle".format(str(batchIndex)))
                self.__savePickleBatches(batchesImages[i], batchesAnnotations[i], imageFile, pickleBatchFile)
                batchIndex += 1

            self.__printProgress(index, len(imageFiles))




