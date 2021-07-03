from django.shortcuts import render
from django.http import HttpResponse

import sys
import os
filePath = os.path.dirname(__file__)
upDir = os.path.split(filePath)
upDir = os.path.split(upDir[0])
upDir = os.path.split(upDir[0])

sys.path.append(upDir[0])
from main import Main
from utils.threading import ThreadWithTrace

main = Main(category="person")

def runTrainThread():
    main.runTrain(lr=0.001)

def initThread():
    threadTrain = ThreadWithTrace(target=runTrainThread)
threadTrain = ThreadWithTrace(target=runTrainThread)

def index(request):
    return HttpResponse("{}".format(sys.path))

def runTrain(request):
    if threadTrain.is_alive():
        response = "Training is running"
    else:
        initThread()
        threadTrain.start()
        response = "Training is started"
    return HttpResponse("{}".format(str(response)))

def stopTrain(request):
    if threadTrain.is_alive():
        threadTrain.kill()
        threadTrain.join()
        response = "Thread stopped"
    else:
        response = "Thread is not running"
    return HttpResponse("{}".format(str(response)))
