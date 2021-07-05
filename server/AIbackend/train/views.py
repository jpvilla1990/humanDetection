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

lr = 0.00001
def writeLr(lr):
    with open("lr.txt", "w") as f:
        f.write(str(lr))

def readLr():
    with open("lr.txt", "r") as f:
        lr = f.readlines

    return float(lr.split("\n")[0])

def runTrainThread():
    lr = readLr()
    main.runTrain(lr=lr)

def initThread():
    del threads[0]
    threads.append(ThreadWithTrace(target=runTrainThread))

threads = [ThreadWithTrace(target=runTrainThread)]

def index(request):
    return HttpResponse("{}".format("Train Index"))

def runTrain(request):
    """
        Method to trigger the training processing receiving as parameter the learning rate
    """
    if request.method == "GET":
        lr_get = float(request.GET["lr"])
        writeLr(lr_get)

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
    imageName = 'loss.jpg'
    # if os.path.exists(imageName):
    #     os.remove(imageName)

    loss = main.getLoss()
    matplotlib.use('Agg')
    plt.plot(loss)
    plt.savefig(imageName)
    plt.close()

    im = Image.open(imageName)

    response = HttpResponse(content_type='image/jpg')
    im.save(response, "JPEG")
    im.close()
    #response['Content-Disposition'] = 'attachment; filename="piece.jpg"'
    return response
