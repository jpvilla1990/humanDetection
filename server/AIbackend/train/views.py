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

def writeLr(lr):
    with open("lr.txt", "w") as f:
        f.write(str(lr))

def readLr():
    with open("lr.txt", "r") as f:
        lr = f.readlines()

    return float(lr[0].split("\n")[0])

def runTrainThread():
    lr = readLr()
    main.runTrain(lr=lr)

def initThread():
    del threads[0]
    threads.append(ThreadWithTrace(target=runTrainThread))

threads = [ThreadWithTrace(target=runTrainThread)]

def index(request):
    return HttpResponse("{}".format("Train Index"))

def startTrain(request):
    """
        Method to trigger the training processing receiving as parameter the learning rate
        ipaddress/startTrain?lr=0.000001
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
    """
        Method to stop training
        ipaddress/stopTrain
    """
    if threads[0].is_alive():
        threads[0].kill()
        threads[0].join()
        response = "Thread stopped"
    else:
        response = "Thread is not running"
    return HttpResponse("{}".format(str(response)))

def getTrainStatus(request):
    """
        Method to get status of training, parameter lines to indicate the number of lines to obtain, and parameter download to indicate if
        the log should be downloaded as a plain text file
        ipaddress/getTrainStatus?lines=50?download=True
    """
    if request.method == "GET":
        lines = int(request.GET["lines"])
        download = bool(request.GET["download"])
        response = []
        if threads[0].is_alive():
            state = "Training Running"
        else:
            state = "Training Stopped"
        response.append(state)
        log = main.getTrainStatus(lines=lines)

        print(download)
        print(type(download))
        if download:
            if os.path.exists("log.txt"):
                os.remove("log.txt")
            for l in log:
                with open("log.txt", "a") as f:
                    f.write("{}\n".format(str(l)))

            response = HttpResponse(open("log.txt", 'rb').read())
            response['Content-Type'] = 'text/plain'
            response['Content-Disposition'] = 'attachment; filename=log.txt'

        else:
            response.append(log)
            response = HttpResponse("{}".format(str(response)))
        return response

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
