from MouseDetector import MouseDetector
import tensorflow as tf
import numpy as np
import time
import cv2 
from datetime import datetime as dt
from numberExtractor import NumberExtractor as ne

inputSource = "video_20211023_133202.mp4"

md = MouseDetector(None, mouseDetection=True)

print("+++++CP1++++")

PATH_TO_MODEL_DIR = "mouseDetectorModel/my_model"
PATH_TO_SAVED_MODEL_MOUSE = PATH_TO_MODEL_DIR + "/saved_model"

show = True

md.build(mouseDetectionModel=PATH_TO_SAVED_MODEL_MOUSE, mask="r")

print("+++++CP2++++")

cap = cv2.VideoCapture(inputSource)

fps = cap.get(cv2.CAP_PROP_FPS)

pFrame = None

numFrames = 60

detectTimes = [20, 10, 5, 3, 2, 1]

process_full = 10

MAX_CENTER_SIZE = 30

MAX_DIFF_SIZE = 60

file_name = ".npy"

times = ["14:00","14:15", "14:30", "14:45","15:00", "15:15", "15:30", "15:45","16:00", "16:15", "16:30", "16:45","17:00", "17:15", "17:30", "17:45","18:00", "18:15", "18:30", "18:45", "19:00", "19:15", "19:30", "19:45", "20:00", "20:15", "20:30", "20:45", "21:00", "21:15", "21:30", "21:45", "22:00", "22:15", "22:30", "22:45", "23:00", "23:15", "23:30", "23:45",]

t1 = "13:32"

dt1 = dt.strptime(t1, "%H:%M")

numExtractor = ne()

#g = 0

for g, x in enumerate(times):


    frames = []
    centers = []
    diffs = []

    count = 0
    c = 0 


    dt2 = dt.strptime(x, "%H:%M")

    time_difference = (dt2-dt1).total_seconds()

    cap.set(cv2.CAP_PROP_POS_FRAMES, time_difference*fps)

    last_time = None

    ret = True

    started = False

    while(ret):
        ret, frame = cap.read()
        nFrame = frame.copy()

        count = count + 1

        if process_full is not None:
            if count % process_full != 0:
                continue

        t = numExtractor.extract(frame)

        t = t[11:16]

        if t != last_time:
            print(t)
            last_time = t
          
        if t in times or (c < 60 and started):

            started = True
                   
            nFrame = md.rectangleMask(nFrame)

            if pFrame is not None:
                diff = cv2.absdiff(nFrame, pFrame).sum()

                diffs.append([count, diff])
                if len(diffs) > 60:
                    diffs.pop(0)

            pFrame = nFrame

            detections = md.mouseDetection(nFrame, md.mouse_detect_fn)

            scores = detections[1]
            detections  = [detections[0]]

            print(scores)

            n_height, n_width, _ = nFrame.shape

            #print(nFrame.shape)
            for i in detections:
                if scores > 0.95:
                    cv2.rectangle(frame, (int(i[1]*n_width), int(i[0]*n_height)), (int(i[3]*n_width), int(i[2]*n_height)), (0, 255, 0 ), 3) 
               
                if show:
                    cv2.imshow("nframe", frame)

                if scores <= 0.95:
                    c = c - 1
                    print("Missed Frame!")

                    continue

                print(c)

                center = md.center(int(i[0]*n_height), int(i[1]*n_width), int(i[2]*n_height) - int(i[0]*n_height), int(i[3]*n_width) - int(i[1]*n_width))
                centers.append([count, center])
                if len(centers) > 30:
                    centers.pop(0)

                nC = frame[int(i[0]*n_height):int(i[2]*n_height),int(i[1]*n_width):int(i[3]*n_width)]

                y_diff = numFrames - c

                if y_diff in detectTimes:
                    nC = frame[int(i[0]*n_height):int(i[2]*n_height),int(i[1]*n_width):int(i[3]*n_width)]
                    nPadded = md.imagePadder(nC, (250, 250))
                    if nPadded is None:
                        c = c - 1
                        continue
                    print("Appended!")
                    frames.append([count, nPadded])
            
            c = c + 1
        

            if show:
                # show one frame at a time
                key = cv2.waitKey(1)
                # Quit when 'q' is pressed
                if key == ord('q'):
                    break

        elif c == 60:
            c = 0
            cv2.destroyAllWindows()
            break
       

    print("Saving!!!")

    carr = np.asarray(centers)
    print(carr.shape)

    darr = np.asarray(diffs)
    print(darr.shape)

    farr = np.asarray(frames)
    print(farr.shape)

    ndict = {"centers": carr, "diffs":darr, "frames":farr}

    sora = input("Was this sleep or awake?")

    with open("extractions/" + str(sora) + str(g) + file_name,'wb') as f:
        np.save(f, ndict)

    print("DONE!")

cap.release()
cv2.destroyAllWindows()

        