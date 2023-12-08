from concurrent.futures import process
import numpy as np
import cv2
from math import sqrt
import pandas as pd
#from models.workspace.MouseDetection.test import MouseDetection
import tensorflow as tf
import time


def mouse_callback(event, x, y, flags, params):

    #right-click event value is 2
    if event == 2:
        print(f"{x}, {y}")

class MouseDetector:
    """Functions for Mouse Object Detection and Behavior Analysis"""

    def __init__(self, m_inputSource = None, mouseDetection = False, earDetection=False, behaviorDetection=False) -> None:
        self.cap = cv2.VideoCapture(m_inputSource)
        self.m_earDetection = earDetection
        self.m_behaviorDetection = behaviorDetection
        self.m_mouseDetection = mouseDetection
        self.MAX_LENGTH = 300
        self.MIN_LENGTH = 100
        self.DISTANCE_THRESHOLD = 200
        self.OFFSET = 15
        self.ear_detect_fn = None
        self.ear_detect_label = None
        self.behavior_detect_fn = None
        self.behavior_detect_label = None
        self.mouse_detect_fn = None
        self.mouse_detect_label = None
        self.behavior_buffer = []
        self.position_buffer = []
        self.fgbg = None
        self.maskType = None
        self.circle_img = None
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def build(self, mouseDetectionModel=None, mouseDetectionLabels=None, earDetectionModel=None, earDetectionLabels=None, behaviorDetectionModel=None, behaviorDetectionLabels=None, mask=None):
        #self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG() 
        self.maskType = mask

        if self.m_earDetection == True:
            self.ear_detect_label = {1: {'id': 1, 'name': 'ear'}}

            print('Loading model...', end='')
            start_time = time.time()

            # Load saved model and build the detection function
            self.ear_detect_fn = tf.saved_model.load(earDetectionModel)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print('Done! Took {} seconds'.format(elapsed_time))

        if self.m_mouseDetection == True:
            self.mouse_detect_label = {1: {'id': 1, 'name': 'mouse'}}

            print('Loading model...', end='')
            start_time = time.time()

            # Load saved model and build the detection function
            self.mouse_detect_fn = tf.saved_model.load(mouseDetectionModel)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print('Done! Took {} seconds'.format(elapsed_time))
        
        if self.m_behaviorDetection == True:
            print('Loading behavior detection model...', end='')
            start_time = time.time()

            self.behavior_detect_fn = tf.keras.models.load_model(behaviorDetectionModel)

            end_time = time.time()
            elapsed_time = end_time - start_time
            
            self.behavior_detect_label = behaviorDetectionLabels

            print('Done! Took {} seconds'.format(elapsed_time))
        
    def backgroundSubtractor(self, frame):
        height,width, c_channels = frame.shape

        # Background Subtractor
        mask = np.zeros((height,width, c_channels), np.uint8)
        n_frame = cv2.GaussianBlur(frame, (5,5), cv2.BORDER_DEFAULT)
        fgmask = self.fgbg.apply(n_frame)
        ret,thresh = cv2.threshold(fgmask,127,255,0)
        contours, heirarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        

        if len(contours) != 0:
            # draw in blue the contours that were founded
            # find the biggest countour (c) by the area
            c = max(contours, key = cv2.contourArea)
            return c

        return None

    def contourDetector(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, (20,20))
        
        ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

        edges = cv2.Canny(thresh, 550, 450)

        contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key = cv2.contourArea)

        return c

    def behaviorAnalysis(self, img1, img2, img3, centers):
        height,width, c_channels = img1.shape

        #Run the behavior analysis
        img1 = tf.expand_dims(img1, axis=0)
        img2 = tf.expand_dims(img2, axis=0)
        img3 = tf.expand_dims(img3, axis=0)
        centers = tf.expand_dims(centers, axis=0)

        predictions = self.behavior_detect_fn.predict([img1, img2, img3, centers])
        
        prediction = self.behavior_detect_label[np.argmax(predictions)]

        return prediction

    def earDetection(self, image, model):
        height,width, c_channels = image.shape

        #Run the ear detection
        n_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        input_tensor = tf.convert_to_tensor(n_image)
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = model(input_tensor)
        
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        topDetections = [detections['detection_boxes'][0], detections['detection_boxes'][1]]
                
        return topDetections

    def mouseDetection(self, image, model = None):

        if model == None:
            model = self.mouse_detect_fn
        height,width, c_channels = image.shape

        #Run the ear detection
        n_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        input_tensor = tf.convert_to_tensor(n_image)
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = model(input_tensor)
        
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        topDetections = [detections['detection_boxes'][0], detections['detection_scores'][0]]
                
        return topDetections
    
    def run(self, mouseDetector=False, backgroundSubtractor=False, contourDetector=False, earDetector=False, behaviorAnalysis=False, fileName=None, show=False, timestamp=None, process_full=None):
        centers = []
        timeStamp = 0
        
        while(True):
            if timestamp:
                t = self.timeStamp(currentFrame=timeStamp, timeStamps=timestamp)
                
                if t != -1:
                    print(f"Processing minute {(t / 60)}")

            timeStamp = timeStamp + 1
            ret, frame = self.cap.read()
            if process_full is not None:
                if timeStamp % process_full != 0:
                    continue

            cleanFrame = np.copy(frame)
            if ret == False:
                break

            height,width, c_channels = frame.shape

            #print("Hello")

            if backgroundSubtractor == True:
                c = self.backgroundSubtractor(frame)
                if c is not None:
                    cv2.drawContours(cleanFrame, c, -1, 255, 3)
                    x, y, w, h = self.contourToBox(c)
                    cv2.rectangle(cleanFrame,(x,y),(x+w,y+h),(0,0,255),2)

            if mouseDetector:
                nFrame = frame.copy()
                if self.maskType == "c" or self.maskType == "circle":
                    nFrame = self.circleTransform(nFrame)
                elif self.maskType == "r" or self.maskType == "rectangle":
                    nFrame = self.rectangleMask(nFrame)

                mouseDetect = self.mouseDetection(nFrame, self.mouse_detect_fn)
                n_height, n_width, _ = nFrame.shape
                #print(nFrame.shape)
                for i in mouseDetect:
                    if self.goodDetection(int(i[2]*n_height) - int(i[0]*n_height), int(i[3]*n_width) - int(i[1]*n_width), 200, 40):
                        pass 
                    else:
                        break
                    cv2.rectangle(nFrame, (int(i[1]*n_width), int(i[0]*n_height)), (int(i[3]*n_width), int(i[2]*n_height)), (0, 255, 0 ), 3) 
                    if show:
                        cv2.imshow("nframe", nFrame)
                    cv2.setMouseCallback('nframe', mouse_callback)
                    center = self.center(int(i[0]*n_height), int(i[1]*n_width), int(i[2]*n_height) - int(i[0]*n_height), int(i[3]*n_width) - int(i[1]*n_width))
                    centers.append([timeStamp / self.fps, center])

                    nC = frame[int(i[0]*n_height):int(i[2]*n_height),int(i[1]*n_width):int(i[3]*n_width)]
                    nCropped = nC.copy()

                    c = self.contourDetector(nCropped)
                    if c is not None:
                        cv2.drawContours(nCropped, c, -1, 255, 3)
                        x, y, w, h = self.contourToBox(c)

                        nnCropped = nCropped[y:y+h,x:x+w]

                        if earDetector:
                            
                            earDetect = self.earDetection(nCropped, self.ear_detect_fn)

                            n_height, n_width, _ = nCropped.shape

                            for i in earDetect:
                                
                                cv2.rectangle(nCropped, (int(i[1]*n_width), int(i[0]*n_height)), (int(i[3]*n_width), int(i[2]*n_height)), (0, 255, 0), 3)
                            if show:
                                cv2.imshow("nCropped", nCropped)
                    
                        if behaviorAnalysis:
                            nPadded = self.imagePadder(nC, (200, 200))
                            cv2.imshow("nbframe", nnCropped)
                            self.behavior_buffer.append(nPadded)
                            self.position_buffer.append(center)

                            if len(self.behavior_buffer) > 5:
                                self.behavior_buffer.pop(0)
                            if len(self.position_buffer) > 15:
                                self.position_buffer.pop(0)

                            if len(self.behavior_buffer) == 5 and len(self.position_buffer) == 15:
                                np_img1 = np.array(self.behavior_buffer)
                                np_centers = np.array(self.position_buffer)
                                tf_img1 = tf.convert_to_tensor(np_img1)
                                tf_centers = tf.convert_to_tensor(np_centers)
                                detectedBehavior = self.behaviorAnalysis(tf_img1[0], tf_img1[2], tf_img1[4], tf_centers)
                                print(f"Detected Behavior {detectedBehavior}")

            if contourDetector == True:
                nFrame = frame.copy()

                if self.maskType == "c" or self.maskType == "circle":
                    nFrame = self.circleTransform(frame)
                elif self.maskType == "r" or self.maskType == "rectangle":
                    nFrame = self.rectangleMask(frame)
                if show:
                    cv2.imshow("nFrame", nFrame)

                c = self.contourDetector(nFrame)
                if c is not None:
                    cv2.drawContours(cleanFrame, c, -1, 255, 3)
                    x, y, w, h = self.contourToBox(c)
                    cv2.rectangle(cleanFrame,(x,y),(x+w,y+h),(0,255,0),2)

                if earDetector == True:
                    n_x, n_y, n_w, n_h = self.offsetAdder(x, y, w, h, self.OFFSET)
                    nCropped = frame[n_y:n_y+n_h, n_x:n_x+n_w]
                    
                    earDetect = self.earDetection(nCropped, self.ear_detect_fn)

                    n_width, n_height, _ = nCropped.shape

                    for i in earDetect:
                        c = self.center(i[1] * n_height, i[0] * n_width, (i[3] * n_height) - (i[1] * n_height), (i[2] * n_width) - (i[0] * n_width))
                        cv2.circle(nCropped, (int(c[0]), int(c[1])), 6, (0, 255,0), -1) 

                    if show:
                        cv2.imshow("nFrame", nCropped)

                    cleanFrame[n_y:n_y+n_h, n_x:n_x+n_w] = nCropped

            
                    
            #cv2.imshow("frame", cleanFrame)
            if show:

                cv2.imshow("frame", cleanFrame)
                # show one frame at a time
                key = cv2.waitKey(1)
                # Quit when 'q' is pressed
                if key == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

        print("Saving!!!")

        narr = np.asarray(centers)
        with open(fileName,'wb') as f:
            np.save(f, narr)

        print("DONE!")

    def contourToBox(self, contour):
        return cv2.boundingRect(contour)

    def offsetAdder(self, x, y, w, h, n_offset):
        #Add an offset to Image
        n_x = x - n_offset
        n_y = y - n_offset
        n_w = w + n_offset*2
        n_h = h + n_offset*2

        return n_x, n_y, n_w, n_h

    def distance(self, p1, p2):
        # Return distance between two points
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def center(self, x, y, w, h):
        # Return center of box
        center = []
        center.append(x + (w / 2))
        center.append(y - (h / 2))
        return center

    def imagePadder(self, image, dimensions):
        #Pad images to correct shape
        w, h, _ = image.shape

        n_w, n_h = dimensions

        if w > n_w or h > n_h:
            print("TOO BIG!!!!")
            return None

        n_image = cv2.copyMakeBorder(image, int((n_w - w) / 2), int((n_w - w) / 2), int((n_h - h) / 2), int((n_h - h) / 2), cv2.BORDER_CONSTANT)
        
        if n_image.shape[0] == dimensions[0] - 1:
            n_image = cv2.copyMakeBorder(n_image, 0, 1, 0, 0, cv2.BORDER_CONSTANT)
 
        if n_image.shape[1] == dimensions[0] - 1:
            n_image = cv2.copyMakeBorder(n_image, 0, 0, 1, 0, cv2.BORDER_CONSTANT)
           
        if n_image.shape[0] == dimensions[1] - 1:
            n_image = cv2.copyMakeBorder(n_image, 1, 0, 0, 0, cv2.BORDER_CONSTANT)
           
        if n_image.shape[1] == dimensions[1] - 1:
            n_image = cv2.copyMakeBorder(n_image, 0, 0, 0, 1, cv2.BORDER_CONSTANT)

        return n_image

    def circleTransform(self, frame):
        height,width, c_channels = frame.shape

        # Hough Circle Transform
        mask = np.zeros((height,width,c_channels), np.uint8)
        if self.circle_img is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

            edges = cv2.Canny(gray, 100, 200)
            circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=250,maxRadius=0)
            print(circles.shape)
            for i in circles:
                m_circle = i[0]    

                i = m_circle
                self.circle_img = cv2.circle(mask,(int(i[0]),int(i[1]) + 60),int(i[2]),(255,255,255),thickness=-1)
        
        masked_data = cv2.bitwise_and(frame, self.circle_img)   
        masked_data[np.where((masked_data==[0,0,0]).all(axis=2))] = [255,255,255] 

        return masked_data

    def rectangleMask(self, frame, x = 360, y = 190, x2 = 874, y2 = 1000):
        height,width, c_channels = frame.shape

        mask = np.zeros((height,width,c_channels), np.uint8)

        rectangleImg = cv2.rectangle(mask, (x, y), (x2, y2),(255,255,255), thickness=-1)

        masked_data = cv2.bitwise_and(frame, rectangleImg)   
        masked_data[np.where((masked_data==[0,0,0]).all(axis=2))] = [255,255,255]

        return masked_data 

    def consistentMotion(self, c1, c2, maxDistance):
        if self.distance(c1, c2) > maxDistance:
            return False
        else:
            return True
            
    def timeStamp(self, currentFrame, timeStamps):
        if currentFrame % (self.fps * timeStamps) == 0:
            return currentFrame / self.fps
        else:
            return -1

    def goodDetection(self, w, h , max, min):
        if w < max and h < max:
            if w > min and h > min:
                return True
        return False