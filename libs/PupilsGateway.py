import cv2
from libs import FaceGateway
from matplotlib import pyplot as plt

def blob_process(eye, detector, threshold, static):
    gray_frame = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2) #1
    img = cv2.dilate(img, None, iterations=4) #2
    img = cv2.medianBlur(img, 5) #3
    keypoints = detector.detect(img) #passa a imagem para dentro do detetor para detetar a pupila

    if static:
      plt.axis("off")
      plt.subplot(2,2,1)
      plt.imshow(gray_frame, cmap='gray')
      #plt.show()

      plt.axis("off")
      plt.subplot(2,2,2)
      plt.imshow(img, cmap='gray')
      plt.show()

    return keypoints

def detect_pupil(img, threshold, face_coordinates, eyes, static):

  detector_params = cv2.SimpleBlobDetector_Params()
  detector_params.filterByArea = True
  detector_params.maxArea = 1500
  blobDetector = cv2.SimpleBlobDetector_create(detector_params)

  face = FaceGateway.calculateFace(img, face_coordinates)

  for (ex,ey,ew,eh) in eyes:
      eye = face[ey:ey + eh, ex:ex + ew]
      keypoints = blob_process(eye, blobDetector, threshold, static)
      cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)