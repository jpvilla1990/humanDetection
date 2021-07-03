from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('runTrain', views.runTrain, name='runTrain'),
    path('stopTrain', views.stopTrain, name='stopTrain'),
    path('getTrainStatus', views.getTrainStatus, name='getTrainStatus'),
    path('stopServer', views.stopServer, name='stopServer'),
]