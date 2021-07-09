from django.shortcuts import render
from django.http import HttpResponse
import urllib.request

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
        ipaddress/getTrainStatus?lines=50&download=True
    """
    if request.method == "GET":
        lines = int(request.GET["lines"])
        download = request.GET["download"]
        response = []
        if threads[0].is_alive():
            state = "Training Running"
        else:
            state = "Training Stopped"
        response.append(state)
        log = main.getTrainStatus(lines=lines)

        print(download)
        print(type(download))
        if download == "True":
            if os.path.exists("log.txt"):
                os.remove("log.txt")
            for l in log:
                with open("log.txt", "a") as f:
                    f.write("{}\n".format(str(l)))

            response = HttpResponse(open("log.txt", 'rb').read())
            response['Content-Type'] = 'text/plain'
            response['Content-Disposition'] = 'attachment; filename=log.txt'

        elif download == "False":
            response.append(log)
            response = HttpResponse("{}".format(str(response)))
        return response

def getLoss(request):
    """
        Method to obtain the loss either downloading the image or either rendering the image
        ipaddress/getLoss?download=True&image=True
    """
    if request.method == "GET":
        image = request.GET["image"]
        download = request.GET["download"]
        imageName = 'loss.jpg'
        lossFile = "loss.txt"
        if os.path.exists(lossFile):
            os.remove(lossFile)

        loss = main.getLoss()

        if image == "True":
            matplotlib.use('Agg')
            plt.plot(loss)
            plt.savefig(imageName)
            plt.close()

            im = Image.open(imageName)

            response = HttpResponse(content_type='image/jpg')
            im.save(response, "JPEG")
            im.close()

            if download == "True":
                response['Content-Disposition'] = 'attachment; filename="loss.jpg"'
        else:
            for l in loss:
                with open(lossFile, "a") as f:
                    f.write("{}\n".format(str(l)))
            response = HttpResponse(open(lossFile, 'rb').read())

            if download == "True":
                response['Content-Type'] = 'text/plain'
                response['Content-Disposition'] = 'attachment; filename=loss.txt'

    return response

def predictImage(request):
    """
        Method to obtain the a prediction either by an URL image or uploading the image itself
        ipaddress/predictImage?imageURL=url&image=image.jpg&download=True
        http://192.168.178.87:8000/train/predictImage?imageURL=https://res.cloudinary.com/upskilled/image/fetch/w_660,h_440,c_crop,c_fill,g_face:auto,f_auto/https://www.upskilled.edu.au/getmedia%2F69cf3c32-e268-469e-887c-5e152369a5bd%2Fhero-image-people-persons.jpg%3B.aspx%3Fwidth%3D1000%26height%3D667%26ext%3D.jpg&image=image.jpg&download=False
    """
    dirpath = os.path.dirname(__file__)
    if request.method == "GET":
        imageURL = request.GET["imageURL"].split('"')[1]
        image = request.GET["image"]
        download = request.GET["download"]

        imageName = os.path.join(dirpath, 'sample.jpg')
        if os.path.exists(imageName):
            os.remove(imageName)

        opener=urllib.request.build_opener()
        opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(imageURL, imageName)

        im = main.runPrediction(imageName)

        response = HttpResponse(content_type='image/jpg')
        im.save(response, "JPEG")
        im.close()

        if download == "True":
            response['Content-Disposition'] = 'attachment; filename="loss.jpg"'

    return response
