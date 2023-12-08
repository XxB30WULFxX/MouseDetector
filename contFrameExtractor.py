import tensorflow as tf
import numpy as np
import time
from MouseDetector import MouseDetector
import cv2
from datetime import datetime as dt
from numberExtractor import NumberExtractor as ne
import os

SHOW = True

SHOW_INDICATOR = True

TIME_STAMP = 5

ROOT = input("Video Folder?")

videos = [x for x in os.listdir(ROOT) if x[-1] == "4"]

file_name = ".npy"

numExtractor = ne("Numbers")

#Mouse Detector Initialization

md = MouseDetector(None, mouseDetection=True)

PATH_TO_MODEL_DIR = "mouseDetectorModel/my_model"
PATH_TO_SAVED_MODEL_MOUSE = PATH_TO_MODEL_DIR + "/saved_model"

md.build(mouseDetectionModel=PATH_TO_SAVED_MODEL_MOUSE, mask="r")

for x in videos:

    # CV2 setup
    inputSource = os.path.join(ROOT, x)

    cap = cv2.VideoCapture(inputSource)
    fps = cap.get(cv2.CAP_PROP_FPS)

    def average(l):
        return sum(l)/len(l)


    detectTimes = [20, 10, 5, 3, 2, 1]

    numFrames = 60

    count = 0

    predictions = []

    g = 0

    data = []

    class SAanalysis:
        def __init__(self, model, diff_len, centers_len, frames_len):
            self.diffs_len = diff_len
            self.centers_len = centers_len
            self.frames_len = frames_len

            self.model = model

            self.diffs = []
            self.centers = []
            self.frames = []

        def diffAppend(self, value):
            if len(self.diffs) < self.diffs_len:
                self.diffs.append(value)
            elif len(self.diffs) == self.diffs_len:
                self.diffs.pop(0)
                self.diffs.append(value)

        def centerAppend(self, value):
            if len(self.centers) < self.centers_len:
                self.centers.append(value)
            elif len(self.centers) == self.centers_len:
                self.centers.pop(0)
                self.centers.append(value)

        def framesAppend(self, value):
            if len(self.frames) < self.frames_len:
                self.frames.append(value)
            elif len(self.frames) == self.frames_len:
                self.frames.pop(0)
                self.frames.append(value)

        def ready(self):
            if len(self.diffs) == self.diffs_len and len(self.centers) == self.centers_len and len(self.frames) == self.frames_len:
                return True
            else: 
                return False

        def clear(self):
            self.diffs = []
            self.centers = []
            self.frames = []

        def inference(self):
            diffs_avg = average(self.diffs)

            n_center = []
            for i,k in zip(self.centers[0::], self.centers[1::]):
                d = md.distance(i,k)
                n_center.append(d)
                
            centers_avg = average(n_center)    

            t_diffs = tf.convert_to_tensor([diffs_avg], dtype=tf.float32)
            t_centers = tf.convert_to_tensor([centers_avg], dtype=tf.float32)

            t_frames = [tf.image.per_image_standardization(tf.convert_to_tensor(frame, dtype=tf.float32)) for frame in self.frames]

            return self.model([t_frames[0][tf.newaxis, ...],t_frames[1][tf.newaxis, ...],t_frames[2][tf.newaxis, ...],t_frames[3][tf.newaxis, ...],t_frames[4][tf.newaxis, ...],t_frames[5][tf.newaxis, ...], t_diffs[tf.newaxis, ...], t_centers[tf.newaxis, ...]])

    sa_analyser = SAanalysis(None, 60, 30, 6)

    pFrame = None

    c = 1

    ret = True


    while(ret):
        ret, frame = cap.read()

        if ret == False:
            print("Video Done!")
            break

        count += 1

        if TIME_STAMP:
            if count % (TIME_STAMP*60*fps) == 0:
                print(f"Processing Min {count / (60*fps)}")
                print(f"{len(sa_analyser.diffs)} {len(sa_analyser.centers)} {len(sa_analyser.frames)}")
        if count % fps != 0:
            continue

        nFrame = md.rectangleMask(frame, x = 360, y = 120, x2 = 875,y2 = 650)

        cv2.imshow("tstnFrame",nFrame)

        if pFrame is not None:
            diff = cv2.absdiff(nFrame, pFrame).sum()
            sa_analyser.diffAppend([count, diff])
        
        pFrame = nFrame.copy()

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
                #print("Missed Frame!")

                continue

            center = md.center(int(i[0]*n_height), int(i[1]*n_width), int(i[2]*n_height) - int(i[0]*n_height), int(i[3]*n_width) - int(i[1]*n_width))
            sa_analyser.centerAppend([count, center])

            nC = frame[int(i[0]*n_height):int(i[2]*n_height),int(i[1]*n_width):int(i[3]*n_width)]

            y_diff = numFrames - c

            if y_diff in detectTimes:
                nC = frame[int(i[0]*n_height):int(i[2]*n_height),int(i[1]*n_width):int(i[3]*n_width)]
                nPadded = md.imagePadder(nC, (250, 250))
                if nPadded is None:
                    c = c - 1
                    continue

                sa_analyser.framesAppend([count, nPadded])

        c += 1

        if SHOW:
            cv2.imshow("nFrame", frame)

        if sa_analyser.ready():
            print("Saving!!!")
            t = numExtractor.extract(frame)

            t = t[11:16]

            ndict = {"time": t, "frame": count, "centers": sa_analyser.centers, "diffs":sa_analyser.diffs, "frames":sa_analyser.frames}

            data.append(ndict)

            sa_analyser.clear()

            g += 1
            c = 0


        if SHOW:
            # show one frame at a time
            key = cv2.waitKey(1)
            # Quit when 'q' is pressed
            if key == ord('q'):
                break

    with open(f"extractions/{x[:-4]}{file_name}",'wb') as f:
                np.save(f, data)

    cap.release()
    cv2.destroyAllWindows()