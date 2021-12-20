import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
from libs import FaceGateway
from libs import EyesGateway
from matplotlib import pyplot as plt


def detectEyeDirection(eyes, threshold, img, face_coordinates):
    ex,ey,ew,eh = eyes[0]
    face = FaceGateway.calculateFace(img, face_coordinates)
    eye = face[ey:ey + eh, ex:ex + ew]
    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    _, threshold_eye = cv2.threshold(gray_eye, threshold, 255, cv2.THRESH_BINARY)

    height, width = threshold_eye.shape

    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    gaze_ratio = left_side_white / right_side_white #We divide the white pixels of the left part and those of the right part and we get the gaze ratio.

    if gaze_ratio <= 1:
        cv2.putText(img, "RIGHT ", (20, 20), font, 0.5, (0, 0, 255), 1)
    elif 1 < gaze_ratio < 1.4:
        cv2.putText(img, "CENTER ", (20, 20), font, 0.5, (0, 0, 255), 1)
    else:
        cv2.putText(img, "LEFT ", (20, 20), font, 0.5, (0, 0, 255), 1)