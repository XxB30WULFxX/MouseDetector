import tensorflow as tf
import numpy as np
import time
from MouseDetector import MouseDetector
import cv2
from datetime import datetime as dt

SHOW = False

SHOW_INDICATOR = True

TIME_STAMP = 5

file_name = ".npy"


#Mouse Detector Initialization

md = MouseDetector(None, mouseDetection=True)

PATH_TO_MODEL_DIR = "mouseDetectorModel/my_model"
PATH_TO_SAVED_MODEL_MOUSE = PATH_TO_MODEL_DIR + "/saved_model"

md.build(mouseDetectionModel=PATH_TO_SAVED_MODEL_MOUSE, mask="r")

# CV2 setup
inputSource = input("Input Video?")
cap = cv2.VideoCapture(inputSource)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"fps:{fps}")

def average(l):
    return sum(l)/len(l)


pFrame = None

c = 1

ret= True

count = 0

centers_list = []

while(ret):
    ret, frame = cap.read()

    count += 1

    if TIME_STAMP:
        if count % (TIME_STAMP*60*fps) == 0:
            print(f"Processing Min {count / (60*fps)}")
    if count % fps != 0:
        continue

    nFrame = md.rectangleMask(frame, x = 350, y = 110, x2 = 865,y2 = 640)

    if SHOW:
        cv2.imshow("nFrame", nFrame)


    detections = md.mouseDetection(nFrame, md.mouse_detect_fn)

    scores = detections[1]
    detections  = [detections[0]]

    n_height, n_width, _ = nFrame.shape

    #print(nFrame.shape)
    for i in detections:
        if scores > 0.95:
            cv2.rectangle(frame, (int(i[1]*n_width), int(i[0]*n_height)), (int(i[3]*n_width), int(i[2]*n_height)), (0, 255, 0 ), 3) 

        if scores <= 0.95:
            c = c - 1
            print("Missed Frame!")

            continue

        center = md.center(int(i[0]*n_height), int(i[1]*n_width), int(i[2]*n_height) - int(i[0]*n_height), int(i[3]*n_width) - int(i[1]*n_width))
        centers_list.append([count, center])
        


    if SHOW:
        # show one frame at a time
        key = cv2.waitKey(1)
        # Quit when 'q' is pressed
        if key == ord('q'):
            break


print("Saving!!!")

centers = np.asarray(centers_list)

with open(f"centers_{inputSource}_{file_name}",'wb') as f:
    np.save(f, centers)



cap.release()
cv2.destroyAllWindows()