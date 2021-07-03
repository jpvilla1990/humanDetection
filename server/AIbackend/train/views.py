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
    main.runTrain(lr=0.00001)

def initThread():
    del threads[0]
    threads.append(ThreadWithTrace(target=runTrainThread))

threads = [ThreadWithTrace(target=runTrainThread)]

def index(request):
    return HttpResponse("{}".format("Miki I love you so Much!!"))

def runTrain(request):
    if threads[0].is_alive():
        response = "Training is running"
    else:
        initThread()
        threads[0].start()
        response = "Training is started"
    return HttpResponse("{}".format(str(response)))

def stopTrain(request):
    if threads[0].is_alive():
        threads[0].kill()
        threads[0].join()
        response = "Thread stopped"
    else:
        response = "Thread is not running"
    return HttpResponse("{}".format(str(response)))
