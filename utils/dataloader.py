import os
import shutil
import urllib
import zipfile
import json
from collections import defaultdict

from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

class Dataloader(object):
    
    def __init__(self, category='person'):
        self.__cocoTrainUrl = "http://images.cocodataset.org/zips/train2017.zip"
        self.__cocoValUrl = "http://images.cocodataset.org/zips/val2017.zip"
        self.__cocoTestnUrl = "http://images.cocodataset.org/zips/test2017.zip"

        self.__cocoAnnotations = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

        self.__createPaths(category)

        self.organizedDatasetByCategory(category)

    def __createFolder(self, folder):
        """
            Method to create a folder
        """
        try:
            os.mkdir(folder)
        except:
            pass

    def __createPaths(self, category):
        """
            Method to create all needed paths
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

        self.__createFolder(self.__datasetfolder)
        self.__createFolder(self.__train)
        self.__createFolder(self.__val)
        self.__createFolder(self.__test)
        self.__createFolder(self.__annotations)
        self.__createFolder(self.__trainPersons)
        self.__createFolder(self.__trainPersonsImages)
        self.__createFolder(self.__trainPersonsAnn)

    def  __downloadBarProgress(self, url, output):
        """
            Method to download and display progress bar
        """
        with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output, reporthook=t.update_to)

    def __unzipFile(self, inputFile, outputFile):
        """
            Method to unzip a zip file
        """
        with zipfile.ZipFile(inputFile, 'r') as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc='Extracting '):
                try:
                    zip_ref.extract(member, outputFile)
                except zipfile.error as e:
                    print("Error" + str(e))
        os.remove(inputFile)

    def downloadCocoDataset(self):
        """
            Method to download cocoDataset
        """
        trainZip = os.path.join(self.__train, "train.zip")
        valZip = os.path.join(self.__val, "val.zip")
        testZip = os.path.join(self.__test, "test.zip")
        annotationsZip = os.path.join(self.__annotations, "annotations.zip")

        if not os.listdir(self.__train):
            print("Downloading COCO train dataset")
            self.__downloadBarProgress(self.__cocoTrainUrl, trainZip)
            self.__unzipFile(trainZip, self.__train)
        else:
            print("COCO train dataset already downloaded")

        if not os.listdir(self.__val):
            print("Downloading COCO val dataset")
            self.__downloadBarProgress(self.__cocoValUrl, valZip)
            self.__unzipFile(valZip, self.__val)
        else:
            print("COCO val dataset already downloaded")

        if not os.listdir(self.__test):
            print("Downloading COCO test dataset")
            self.__downloadBarProgress(self.__cocoTestnUrl, testZip)
            self.__unzipFile(testZip, self.__test)
        else:
            print("COCO test dataset already downloaded")

        if not os.listdir(self.__annotations):
            print("Downloading COCO annotations dataset")
            self.__downloadBarProgress(self.__cocoAnnotations, annotationsZip)
            self.__unzipFile(annotationsZip, self.__annotations)
        else:
            print("COCO annotations dataset already downloaded")

    def __loadAnnotationsJSONFile(self, file):
        """
            Methot to load json file containing the annotations
        """
        annotations = json.load(open(file, 'r'))
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in annotations:
            for ann in annotations['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in annotations:
            for img in annotations['images']:
                imgs[img['id']] = img

        if 'categories' in annotations:
            for cat in annotations['categories']:
                cats[cat['id']] = cat

        if 'annotations' in annotations and 'categories' in annotations:
            for ann in annotations['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        return anns, imgToAnns, catToImgs, imgs, cats

    def __findCategoryId(self, category, categories):
        """
            Method to return the category id
        """
        for id in categories:
            if categories[id]['name'] == category:
                id_category = id
                break
        return id_category

    def __getAnnotationsByCategory(self, id_category, annotations):
        """
            Method to obtain relevant annotations by category
        """
        filtered_annotations = {}
        for ann in annotations:
            if annotations[ann]['category_id'] == id_category and annotations[ann]['iscrowd'] == 0:
                image_id = annotations[ann]['image_id']
                if image_id in filtered_annotations:
                    filtered_annotations[image_id].append(annotations[ann])
                else:
                    filtered_annotations.update({image_id:[annotations[ann]]})

        return filtered_annotations

    def __getImagesByCategory(self, images, filtered_annotations):
        """
            Method to obtain relevant annotations by category
        """
        filtered_images = {}
        for image_id in filtered_annotations:
            filtered_images.update({image_id:images[image_id]})

        return filtered_images

    def __obtainMasks(self, category, train):
        """
            Method to obtain masks from images and annotations JSON file
            it creates the mask and organize the training set by categories exm: 'person'
        """
        if train:
            loadedFromJSON = self.__loadAnnotationsJSONFile(self.__annotationsFileTrainJSON)
        else:
            loadedFromJSON = self.__loadAnnotationsJSONFile(self.__annotationsFileValJSON)

        annotations = loadedFromJSON[0]
        images = loadedFromJSON[3]
        categories = loadedFromJSON[4]

        id_category = self.__findCategoryId(category, categories)
        filteredAnnotations = self.__getAnnotationsByCategory(id_category, annotations)
        filteredImages = self.__getImagesByCategory(images, filteredAnnotations)

        return filteredAnnotations, filteredImages

    def __createMask(self, annotation, image):
        """
            Method to create mask from annotation
        """
        numberObjects = len(annotation)

        imageSize = image['width'], image['height']
        newImage = Image.new("1", (imageSize[0], imageSize[1]))

        draw = ImageDraw.Draw(newImage)

        for n in range(numberObjects):
            numberSegmentation = len(annotation[n]['segmentation'])
            for m in range(numberSegmentation):
                points = annotation[n]['segmentation'][m]
                draw.polygon((points), fill=1)

        return newImage

    def organizedDatasetByCategory(self, category='person'):
        """
            Method to organize in folders the images and segmentations by category
        """
        if category=='person':
            targetFolderImages = self.__trainPersonsImages
            targetFolderAnnotations = self.__trainPersonsAnn
        else:
            print("Folder for the category " + category + " is not defined")
            exit()

        if not os.listdir(targetFolderImages) or not os.listdir(targetFolderAnnotations):
            pass
        else:
            print("Images and Annotations already created and organized in their respective folders:")
            print(targetFolderImages)
            print(targetFolderAnnotations)
            return

        annotations, images = self.__obtainMasks(category, train=True)

        for image_id in images:
            imageOrigin = os.path.join(self.__imagesTrainFolder, images[image_id]['file_name'])
            imageTarget = os.path.join(targetFolderImages, images[image_id]['file_name'])

            annTarget = os.path.join(targetFolderAnnotations, images[image_id]['file_name'])

            shutil.copyfile(imageOrigin, imageTarget)

            mask = self.__createMask(annotations[image_id], images[image_id])
            mask.save(annTarget)
            mask.close()



            



        


