# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:27:44 2021

@author: Simge
"""
import os 
import cv2
import random
import numpy as np
import pickle

#data direction is "FaceImages" path
DATADIR = "FaceImages"

def get_categories(path):
    my_list = os.listdir(path)
    return my_list

#for each categorized files 
#read images
CATEGORIES = get_categories('FaceImages')

for category in CATEGORIES :
	path = os.path.join(DATADIR, category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

#create list which is empty for training
training_data = []
IMG_SIZE_WEIGHT = 400
IMG_SIZE_HEIGHT = 400

def create_training_data():
	for category in CATEGORIES :
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try :
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array, (IMG_SIZE_WEIGHT, IMG_SIZE_HEIGHT))
				training_data.append([new_array, class_num])
			except Exception as e:
				pass
#function create data for training is called
create_training_data()
#verilen training_data listesindeki
#değerleri karıştırır
random.shuffle(training_data)

X = [] #features
y = [] #labels

#training data içerisindeki her bir feature
#X feature listesine
#her bir label Y listesine eklenir
for features, label in training_data:
	X.append(features)
	y.append(label)

#train dosyasından çekilen öznitelikler
#4 boyuta reshape edilir.
X = np.array(X).reshape(-1, IMG_SIZE_WEIGHT, IMG_SIZE_HEIGHT, 1)
y = np.asarray(y)


# Creating the files containing all the information about model
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
        