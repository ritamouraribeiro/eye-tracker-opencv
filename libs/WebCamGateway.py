import cv2
import time
import sys
import numpy as np
sys.path.append('..')
from libs import FaceGateway
from libs import EyesGateway
from libs import EyeDirectionGateway
from libs import PupilsGateway

#The issue with OpenCV track bars is that they require a function that will happen on each track bar
# movement. We don’t need any sort of action, we only need the value of our track bar, so we create a nothing() function:
def nothing(x):
    pass

def videoCapture(static):

  cap = cv2.VideoCapture(0)
  cv2.namedWindow('image')
  cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
  while True:
      _, img = cap.read()
      face_coordinates = FaceGateway.detectFace(img)
      if np.any(face_coordinates):
          eyes = EyesGateway.detect_eyes(img, face_coordinates)
          if np.any(eyes):
              threshold = cv2.getTrackbarPos('threshold', 'image')
              PupilsGateway.detect_pupil(img, threshold, face_coordinates, eyes, static)
              EyeDirectionGateway.detectEyeDirection(eyes, threshold, img, face_coordinates)
      cv2.imshow("image", img)
      time.sleep(0.1)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  # After the loop release the cap object
  cap.release()
  # Destroy all the windows
  cv2.destroyAllWindows()