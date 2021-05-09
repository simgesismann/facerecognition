# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:41:04 2021

@author: Simge
"""

import cv2
import os
import json 
import tensorflow as tf
import numpy as np


face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
with open('model.json','r') as f:
    model_json = json.load(f)
    
    
model = tf.keras.models.model_from_json(model_json)

model.load_weights('model.h5')
users = list()

for root, dirs, files in os.walk("FaceImages", topdown=False):
    for name in dirs:
        users.append(name)


cap = cv2.VideoCapture(0)
count = 0

while True:
    _,img = cap.read()
    print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        if h>int(img.shape[0])/2 :
            print("face detecting")
            cv2.putText(img,"Bekleyiniz..",(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),3)
            detected_face = img[y-20:y+h+20, x-20:x+w+20]
            resized = cv2.resize(detected_face, (400,400), interpolation = cv2.INTER_AREA)
            normalized= resized/255.
            reshaped=np.reshape(normalized,(3,400,400,1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            print(users[label])
            cv2.putText(img,users[label],(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),3)
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        else:
            cv2.putText(img,"kameraya yaklasin lutfen",(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),3)
    cv2.imshow('img',img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
