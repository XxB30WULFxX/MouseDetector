import tensorflow as tf
import numpy as np
import time
from MouseDetector import MouseDetector
import cv2
from datetime import datetime as dt


def average(l):
    return sum(l)/len(l)

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

    def inference(self):
        diffs_avg = average(self.diffs)

        n_center = []
        for i,k in zip(self.centers[0::], self.centers[1::]):
            d =  MouseDetector.distance(i,k)
            n_center.append(d)
            

        centers_avg = average(n_center)    

        t_diffs = tf.convert_to_tensor([diffs_avg], dtype=tf.float32)
        t_centers = tf.convert_to_tensor([centers_avg], dtype=tf.float32)

        t_frames = [tf.image.per_image_standardization(tf.convert_to_tensor(frame, dtype=tf.float32)) for frame in self.frames]

        return self.model([t_frames[0][tf.newaxis, ...],t_frames[1][tf.newaxis, ...],t_frames[2][tf.newaxis, ...],t_frames[3][tf.newaxis, ...],t_frames[4][tf.newaxis, ...],t_frames[5][tf.newaxis, ...], t_diffs[tf.newaxis, ...], t_centers[tf.newaxis, ...]])

class XY_Behavior_Detection:

    def __init__(self, show=False, show_indicator=True, time_stamp = 5, file_name=".npy"):
        self.SHOW = False

        self.SHOW_INDICATOR = True

        self.TIME_STAMP = 5

        self.file_name = file_name

    def build(self, input_video):

        print('Loading model...', end='')
        start_time = time.time()

        # Load saved model and build the detection function
        self.sa_model = tf.saved_model.load("sa2_saved_model/my_model/")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))

        #Mouse Detector Initialization

        self.md = MouseDetector(None, mouseDetection=True)

        PATH_TO_MODEL_DIR = "mouseDetectorModel/my_model"
        PATH_TO_SAVED_MODEL_MOUSE = PATH_TO_MODEL_DIR + "/saved_model"

        self.md.build(mouseDetectionModel=PATH_TO_SAVED_MODEL_MOUSE, mask="r")

        # CV2 setup
        self.inputSource = input_video
        self.cap = cv2.VideoCapture(self.inputSource)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)


    def run(self):
        detectTimes = [20, 10, 5, 3, 2, 1]

        numFrames = 60

        count = 0

        predictions = []
        centers_list = []


        sa_analyser = SAanalysis(self.sa_model, 60, 30, 6)

        pFrame = None

        c = 1

        ret= True

        latest_predict = None


        while(ret):
            ret, frame = self.cap.read()

            count += 1

            if self.TIME_STAMP:
                if count % (self.TIME_STAMP*60*self.fps) == 0:
                    print(f"Processing Min {count / (60*self.fps)}")
            if count % self.fps != 0:
                continue

            nFrame = self.md.rectangleMask(frame, x = 350, y = 110, x2 = 865,y2 = 640)

            if self.SHOW:
                cv2.imshow("nFrame", nFrame)

            if pFrame is not None:
                diff = cv2.absdiff(nFrame, pFrame).sum()
                sa_analyser.diffAppend(diff)
            
            pFrame = nFrame.copy()

            detections = self.md.mouseDetection(nFrame, self.md.mouse_detect_fn)

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

                center = self.md.center(int(i[0]*n_height), int(i[1]*n_width), int(i[2]*n_height) - int(i[0]*n_height), int(i[3]*n_width) - int(i[1]*n_width))
                centers_list.append([count, center])
                sa_analyser.centerAppend(center)

                nC = frame[int(i[0]*n_height):int(i[2]*n_height),int(i[1]*n_width):int(i[3]*n_width)]

                y_diff = numFrames - c

                if y_diff in detectTimes:
                    nC = frame[int(i[0]*n_height):int(i[2]*n_height),int(i[1]*n_width):int(i[3]*n_width)]
                    nPadded = self.md.imagePadder(nC, (250, 250))
                    if nPadded is None:
                        c = c - 1
                        continue

                    sa_analyser.framesAppend(nPadded)

            c += 1

            if c == numFrames:
                prediction = sa_analyser.inference()
                print(prediction)
                predictions.append([count, prediction])
                c = 1
                latest_predict = float(prediction.numpy())

            if self.SHOW:
                if self.SHOW_INDICATOR and latest_predict is not None:
                    cv2.circle(frame, (150, 150), 20, (255 * (latest_predict - 0.6)*2, 0, 255 - (255*(latest_predict - 0.6)*2)), -1)
                cv2.imshow("Frame", frame)


            if self.SHOW:
                # show one frame at a time
                key = cv2.waitKey(1)
                # Quit when 'q' is pressed
                if key == ord('q'):
                    break


        print("Saving!!!")

        nPredictions = np.asarray(predictions)

        with open(f"predictions_{self.inputSource}_{self.file_name}",'wb') as f:
            np.save(f, nPredictions)

        centers = np.asarray(centers_list)

        with open(f"centers_{self.inputSource}_{self.file_name}",'wb') as f:
            np.save(f, centers)



        self.cap.release()
        cv2.destroyAllWindows()