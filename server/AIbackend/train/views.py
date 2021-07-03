from django.shortcuts import render
from django.http import HttpResponse

import sys
import os
filePath = os.path.dirname(__file__)
upDir = os.path.split(filePath)
upDir = os.path.split(upDir[0])
upDir = os.path.split(upDir[0])

sys.path.append(upDir[0])

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib

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
    return HttpResponse("{}".format("Train Index"))

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

def getTrainStatus(request):
    response = []
    if threads[0].is_alive():
        state = "Training Running"
    else:
        state = "Training Stopped"
    response.append(state)
    response.append(main.getTrainStatus(lines=50))
    return HttpResponse("{}".format(str(response)))

def getLoss(request):
    loss = main.getLoss()
    matplotlib.use('Agg')
    plt.plot(loss)
    plt.savefig('loss.jpg')

    im = Image.open('loss.jpg')

    response = HttpResponse(im, content_type='image/jpg')
    response['Content-Disposition'] = 'attachment; filename="piece.jpg"'
    return response
