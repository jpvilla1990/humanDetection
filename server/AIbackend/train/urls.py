from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('startTrain', views.startTrain, name='startTrain'),
    path('stopTrain', views.stopTrain, name='stopTrain'),
    path('getTrainStatus', views.getTrainStatus, name='getTrainStatus'),
    path('getLoss', views.getLoss, name='getLoss'),
]