import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt
from libs import FaceGateway

def cut_eyebrows(eye):
  height, width = eye.shape[:2]
  eyebrow_h = int(height / 4)
  eye = eye[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
  return eye

def detectEyesWithoutRules(img, face_coordinates):

  img2 = copy.deepcopy(img)

  eye_cascade = cv2.CascadeClassifier('./data/haarcascade_eye.xml')

  gray_face = FaceGateway.calculateGrayFace(img2, face_coordinates)
  face = FaceGateway.calculateFace(img2, face_coordinates) # cut the face frame out

  eyes = eye_cascade.detectMultiScale(gray_face) # detect eyes

  for (ex,ey,ew,eh) in eyes: 
      cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)

  plt.axis("off")    
  plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
  plt.show()

def detect_eyes(img, face_coordinates):
  eye_cascade = cv2.CascadeClassifier('./data/haarcascade_eye.xml')
  
  gray_face = FaceGateway.calculateGrayFace(img, face_coordinates)  
  face = FaceGateway.calculateFace(img, face_coordinates) # cut the face frame out
  
  eyes = eye_cascade.detectMultiScale(gray_face) # detect eyes
  
  width = np.size(face, 1) # get face frame width
  height = np.size(face, 0) # get face frame height
      
  final_eyes = []
      
  for (x, y, w, h) in eyes:
      if y < height/2:
          final_eyes.append([x,y,w,h])

  for (ex,ey,ew,eh) in final_eyes: 
    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,225,255),2)
      
  return final_eyes