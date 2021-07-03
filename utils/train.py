import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
#import matplotlib.pyplot as plt
from torchvision import transforms
from datetime import datetime
import pickle
import random

from utils.model import SwinBlocks
from utils.dataloader import Dataloader
from utils.trainingSetLoader import TrainingSetLoader
from utils.dataPreprocessing import DataPreprocessing

class Train(object):
    """
        Class to perform the training of the NN
    """
    def __init__(self):

        self.__createPaths()
        self.__lastTime = None

        if torch.cuda.is_available():
            self.__cuda = True

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
        self.__logsPath = os.path.join(self.__path, "logs")
        self.__picklePath = os.path.join(self.__path, "pickleFolder")
        self.__pickleModelsPath = os.path.join(self.__picklePath, "pickleModels")

        self.__createFolder(self.__logsPath)
        self.__createFolder(self.__picklePath)
        self.__createFolder(self.__pickleModelsPath)

    def __writeLog(self, logFile, line):
        """
            Method to write log
        """
        logFile = os.path.join(self.__logsPath, logFile)
        file_object = open(logFile, 'a')
        file_object.write("{}\n".format(str(line)))
        file_object.close()

        print(line)

    def __inferenceTime(self):
        """
            Method to obtain the inference time respect last callback
        """
        currentTime = datetime.now()
        if self.__lastTime is None:
            inferenceTime = None
        else:
            inferenceTime = currentTime - self.__lastTime
        self.__lastTime = currentTime

        if inferenceTime:
            return inferenceTime.total_seconds()
        else:
            return inferenceTime

    def __loadPickle(self, file):
        """
            Method to load a pickel file
        """
        with open(file, 'rb') as f:
            return pickle.load(f)

    def __savePickle(self, file, contain):
        """
            Method to load a pickel file
        """
        with open(file, 'wb') as f:
            pickle.dump(contain, f)

    def supervisedLearningTrain(
                                self,
                                category="person",
                                logFile="log.txt",
                                modelFile="supervisedLearning.pickle",
                                batch_size=5,
                                epochs=1000000,
                                model=None,
                                lr=0.001,
                                momentum=0.9,
                                weight_decay=0.0005
                                ):
        """
            Method to train neural network
            using all the datasets
        """
        backUpModelFile = "{}_backUp.pickle".format(modelFile.split(".")[0])
        #trainingSetLoader = TrainingSetLoader(trainingSetFolder)
        dataPreprocessing = DataPreprocessing(
                                              category=category,
                                              batchSize=batch_size
                                             )

        if os.path.isfile(os.path.join(self.__pickleModelsPath, modelFile)):
            try:
                parameters = self.__loadPickle(os.path.join(self.__pickleModelsPath, modelFile))
            except:
                try:
                    parameters = self.__loadPickle(os.path.join(self.__pickleModelsPath, backUpModelFile))
                except:
                    parameters = model.getWeights()
        else:
            parameters = model.getWeights()

        if self.__cuda:
            newParams = parameters
            parameters = []
            for par in newParams:
                parameters.append(par.cuda().detach().requires_grad_())

        optimizer = optim.Adam(parameters, lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        # loss_epoch = []
        # prev_loss_avg = None
        # curr_loss_avg = None
        # prev_loss = None
        # curr_loss = None
        # loss_history = []

        subIndexBatch = 0
        numberSubBatches = 0

        self.__writeLog(logFile, optimizer)

        for epoch in range(epochs):

            #batchFile = trainingSetFolder.getCurrentBatch()
            #batchSample = self.__loadPickle(batchFile)

            if numberSubBatches == subIndexBatch:
                batch = dataPreprocessing.generateBatch()
                subIndexBatch = 0
                imgBatches = batch[0]
                annBatches = batch[1]
                numberSubBatches = len(imgBatches)

            img = imgBatches[subIndexBatch]
            ann = annBatches[subIndexBatch]

            if self.__cuda:
                img = img.cuda()
                ann = ann.cuda()

            subIndexBatch += 1

            optimizer.zero_grad()   # zero the gradient buffers

            img = img.permute(0,2,3,1)

            ann_hat = model.forward(img, parameters)
            ann_hat = torch.flatten(ann_hat, start_dim=1)
            dimX = ann.shape[2]
            dimY = int(ann_hat.shape[1] / dimX)

            ann_hat = ann_hat.reshape([ann.shape[0], ann.shape[1], dimX, dimY])
            ann_hat = torch.nn.functional.interpolate(ann_hat, (ann.shape[2], ann.shape[3]))

            self.__writeLog(logFile, torch.round(torch.sigmoid(ann_hat).sum()))
            self.__writeLog(logFile, ann.sum())
            self.__writeLog(logFile, "Inference Time: {}".format(str(self.__inferenceTime())))
            self.__writeLog(logFile, "Learning Rate: {}".format(str(lr)))

            loss = criterion(ann_hat, ann)

            loss.backward()
            optimizer.step()

                # if (prev_loss is not None) and ((curr_loss > 10 * prev_loss) or (np.isnan(curr_loss.detach().numpy()))):
                #     for layer in model.children():
                #         if hasattr(layer, 'reset_parameters'):
                #             layer.reset_parameters()
                #     for param_group in optimizer.param_groups:
                #         lr_current = lr_current / 10
                #         param_group['lr'] = lr_current
                #     loss_history[len(loss_history) - 1] = prev_loss
                # prev_loss = curr_loss

            if self.__cuda:
                newParams = parameters
                saveParams = []
                for par in newParams:
                    saveParams.append(par.cpu().detach().requires_grad_())
            else:
                saveParams = parameters

            self.__savePickle(os.path.join(self.__pickleModelsPath, modelFile),
                              saveParams)

            if epoch % 10 == 0:
                self.__savePickle(os.path.join(self.__pickleModelsPath, backUpModelFile),
                                  saveParams)
                # plt.plot(step, 10 * np.log10(loss_history))
                # plt.xlabel("batch")
                # plt.ylabel("db")
                # plt.show()
                # np.savetxt(name_model.split(".")[0] + "_loss_history.txt",
                #            np.array(loss_history),
                #            delimiter=',',
                #            fmt="%s")

            self.__writeLog(logFile, "Epoch: {}".format(str(epoch)))
            self.__writeLog(logFile, "Loss: {}".format(str(loss)))
            percentDone = round((epoch / epochs) * 100, 2)
            self.__writeLog(logFile, "Progress: {}".format(str(percentDone)))
            self.__writeLog(logFile, "#################################")
             # print("lr: " + str(lr_current) + ", avg epoch loss: " + str(curr_loss_avg))

            # curr_loss_avg = sum(loss_epoch) / len(loss_epoch)
            # if prev_loss_avg is not None:
            #     if curr_loss_avg > prev_loss_avg:
            #         lr_current = lr_current / 10
            #         for param_group in optimizer.param_groups:
            #             param_group['lr'] = lr_current
            # prev_loss_avg = curr_loss_avg
            # loss_epoch = []

    def returnLog(self, logFile="log.txt", lines=50):
        """
            Method to return last Lines of the log
        """
        linesArray = []
        logFile = os.path.join(self.__logsPath, logFile)

        def read_lastnlines(fname,n):
	    with open(logFile) as f:
		    for line in (f.readlines() [-lines:]):
			    linesArray.append(line)

        return linesArray


