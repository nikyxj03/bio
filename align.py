import os
import cv2
import numpy as np
import pandas as pd
import math
from PIL import Image
from matplotlib import pyplot as plt

def calc_euclideanDistance(m, n):
    x1 = m[0]; y1 = m[1]
    x2 = n[0]; y2 = n[1]

    distance = math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

    return distance

# function to detect faces
def faceDetect(img):
    faces = fdetect.detectMultiScale(img, 1.3, 5)

    if len(faces) > 0:
        face = faces[0]
        x_face, y_face, w_face, h_face = face
        fd_img = img[int(y_face):int(y_face+h_face), int(x_face):int(x_face+w_face)]
        gray_img = cv2.cvtColor(fd_img, cv2.COLOR_BGR2GRAY)

        return fd_img, gray_img

    else:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray_img

# function to align faces
def faceAlign(img_path):
    img1 = cv2.imread(img_path)
    align_img = cv2.imread(img_path)
    # plt.imshow(img1[:, :, ::-1])
    # plt.show()

    img_r = img1.copy()

    fd_img, gray_img = faceDetect(img1) #face detection
    # plt.imshow(fd_img[:, :, ::-1])
    # plt.show()

    eyesdetected = eyedetect.detectMultiScale(gray_img) #eyes detection

    if len(eyesdetected) >= 2:

        baseEyes = eyesdetected[:, 2]
        items = []
        for i in range(0, len(baseEyes)):
            item = (baseEyes[i], i)
            items.append(item)

        df = pd.DataFrame(items, columns = ["length", "idx"]).sort_values(by=['length'], ascending=False)
        eyes = eyesdetected[df.idx.values[0:2]]
        eye1 = eyes[0]; eye2 = eyes[1]

        if eye1[0] < eye2[0]:
            leftEye = eye1
            rightEye = eye2
        else:
            leftEye = eye2
            rightEye = eye1

        center_leftEye = (int(leftEye[0] + (leftEye[2] / 2)), int(leftEye[1] + (leftEye[3] / 2)))
        x_leftEye = center_leftEye[0]; y_leftEye = center_leftEye[1]

        center_rightEye = (int(rightEye[0] + (rightEye[2]/2)), int(rightEye[1] + (rightEye[3]/2)))
        x_rightEye = center_rightEye[0]; y_rightEye = center_rightEye[1]

        # center_of_eyes = (int((x_leftEye+x_rightEye)/2), int((y_leftEye+y_rightEye)/2))

        cv2.circle(fd_img, center_leftEye, 2, (255, 0, 0) , 2)
        cv2.circle(fd_img, center_rightEye, 2, (255, 0, 0) , 2)
        cv2.line(fd_img, center_rightEye, center_leftEye, (67, 67, 67) , 2)

        if y_leftEye > y_rightEye:
            point3 = (x_rightEye, y_leftEye)
            direction = -1
            # print("rotate to clock direction")
        else:
            point3 = (x_leftEye, y_rightEye)
            direction = 1
            # print("rotate to inverse clock direction")

        cv2.circle(fd_img, point3, 2, (255, 0, 0) , 2)

        cv2.line(fd_img,center_rightEye, center_leftEye,(67,67,67),1)
        cv2.line(fd_img,center_leftEye, point3,(67,67,67),1)
        cv2.line(fd_img,center_rightEye, point3,(67,67,67),1)

        a = calc_euclideanDistance(center_leftEye, point3)
        c = calc_euclideanDistance(center_rightEye, point3)
        b = calc_euclideanDistance(center_rightEye, center_leftEye)

        denominator = 2*b*c

        if(denominator > 0):
            cosine_a = (b*b + c*c - a*a)/(2*b*c)
        else:
            cosine_a = 0

        angle = np.arccos(cosine_a)

        angle = (angle * 180) / math.pi

        if direction == -1:
            angle = 90 - angle

        align_img = Image.fromarray(img_r)
        align_img = np.array(align_img.rotate(direction * angle))

    return align_img


opencv_home = cv2.__file__
folders = opencv_home.split(os.path.sep)[0:-1]

path = folders[0]
for folder in folders[1:]:
    path = path + "/" + folder

fd_path = path+"/data/haarcascade_frontalface_default.xml"
eye_detector_path = path+"/data/haarcascade_eye.xml"

fdetect = cv2.CascadeClassifier(fd_path)
eyedetect = cv2.CascadeClassifier(eye_detector_path)

def rotate(image):
    images = [image]

    for instance in images:
        aligned_img = faceAlign(instance)
        # plt.imshow(aligned_img[:, :, ::-1])
        # plt.show()

        final_img, gray_img = faceDetect(aligned_img)
        # plt.imshow(final_img[:, :, ::-1])
        # plt.show()

    return final_img