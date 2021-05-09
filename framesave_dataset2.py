# -*- coding: utf-8 -*-
"""
Created on Mon May  3 22:10:01 2021

@author: Simge
"""

import cv2
import os
#for face detection Haarcascade dataset is used 
#cascade classifier loads frontal_face datas
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#path is the folder named "FaceImages
path = 'FaceImages'
#create function to create folder for each unique ID
def create_folder(user):
    try:
        if not os.path.exists(path + '/' + user):
            os.makedirs(path + '/' + user)
            return True
        else:
            print("User already exist")
            return False
    except OSError:
        print('Error: Creating directory of data')
#create function to save image when face is detected        
def save_image(path, img, i):
    s = "{0}face.jpg"
    s1 = path + '/' + s.format(i)
    cv2.imwrite(s1,img)
#function to open camera and it forms 1280,720 dimensions
def get_face_images(username):
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    count = 0
    
    while True:
        _,img = cap.read()  #img is captured frames from realtima
        #frames will be in grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #to detect face in frame .detectMultiScale method is used
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
        #detectMultiScale returns x,y,w,h values the parameters for rectangle which include detected face
        for (x,y,w,h) in faces:
            if h>int(img.shape[0])/2 :
                #if foung height values is greater than image_y/2 
                print("face detecting")
                cv2.putText(img,"Bekleyiniz..",(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),3)
                #region of interest is cropped from frames according to x and y values
                detected_face = img[y-20:y+h+20, x-20:x+w+20]

                try:
                    #resize the detected images to train model for same sized frames
                    resized = cv2.resize(detected_face, (400,400), interpolation = cv2.INTER_AREA)
                except Exception as e:
                    print(str(e))
                #after resize, save the image with counting
                save_image((path + '/' + username), resized, count)
                #increase count when detected face is found
                count += 1
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            else:
                cv2.putText(img,"kameraya yaklasin lutfen",(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),3)
        cv2.imshow('img',img)
        if count == 100:
            break
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# here we enter username if it s not created 
#create folder with entered username
username = input("Enter username: ")
result = create_folder(username)
#if folder is created then
# get_face_images function is called 
if(result == True):
    get_face_images(username);