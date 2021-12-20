import cv2
import numpy as np

def detectFace(img): #

  face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

  gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#make picture gray
  faces = face_cascade.detectMultiScale(gray_picture, 1.3, 5)

  if np.any(faces):
    for (x,y,w,h) in faces:
      cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

    return faces[0]

  return None

def calculateFace(img, face_coordinates):
  x,y,w,h = face_coordinates
  return img[y:y+h, x:x+w] # cut the face frame out

def calculateGrayFace(img, face_coordinates):
  x,y,w,h = face_coordinates

  gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#make picture gray
  gray_face = gray_picture[y:y+h, x:x+w] # cut the gray face frame out

  return gray_face