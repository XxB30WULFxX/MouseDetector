import numpy as np
from cv2 import cv2
from math import sqrt
import pandas as pd
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

print(tf.__version__)

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

PATH_TO_LABELS = "annotations/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import time
PATH_TO_MODEL_DIR = "exported-models/my_model"
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

cap = cv2.VideoCapture("move5.mp4")

MAX_LENGTH = 300
MIN_LENGTH = 100
DISTANCE_THRESHOLD = 200
OFFSET = 15

m_labels = ["sleeping/still", "running", "fidgeting", "standing", "scratching"]

fgbg = cv2.createBackgroundSubtractorMOG2() 
fgbg2 = cv2.createBackgroundSubtractorMOG2()

print('Loading model2...', end='')
start_time = time.time()

model = tf.keras.models.load_model('my_model')

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))
  
def behaviorAnalysis(img1, img2, img3, centers):
    img1 = tf.expand_dims(img1, axis=0)
    img2 = tf.expand_dims(img2, axis=0)
    img3 = tf.expand_dims(img3, axis=0)
    centers = tf.expand_dims(centers, axis=0)
    predictions = model.predict([img1, img2, img3, centers])
    prediction = m_labels[np.argmax(predictions)]
    print(prediction)
    return prediction

def earDetection(image, model):
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

    image_np_with_detections = n_image.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=2,
            min_score_thresh=.30,
            agnostic_mode=False)
            
    return image_np_with_detections


def offsetAdder(x, y, w, h, n_offset, width, height):
    print(f"width {width}")
    n_x = x - n_offset
    n_y = y - n_offset
    n_w = w + n_offset*2
    n_h = h + n_offset*2

    return n_x, n_y, n_w, n_h


def distance(p1, p2):
    # Return distance between two points
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def center(x, y, w, h):
    # Return center of box
    center = []
    center.append(x + (w / 2))
    center.append(y - (h / 2))
    return center

def createMotionImage(images):
    images[0][np.where((images[0]==[255,255,255]).all(axis=2))] = [0,0,0]
    images[0][np.where((images[0]!=[0,0,0]).all(axis=2))] = [255, 255,255] 
    baseImg = images[0] 
    for i in range(images.shape[0]):
        images[i][np.where((images[i]==[255,255,255]).all(axis=2))] = [0,0,0] 
        images[i][np.where((images[i]!=[0,0,0]).all(axis=2))] = [255 - i*10,255,0 + i *10] 
        img = cv2.bitwise_not(images[i])
        img = cv2.bitwise_and(img, baseImg)
        dst = cv2.bitwise_or(images[i],baseImg)
        baseImg = dst
    return baseImg

def imagePadder(image, dimensions):
    print(f"img shape {image.shape}")
    w, h, _ = image.shape
    n_w, n_h = dimensions
    n_image = cv2.copyMakeBorder(image, int((n_w - w) / 2), int((n_w - w) / 2), int((n_h - h) / 2), int((n_h - h) / 2), cv2.BORDER_CONSTANT)
    print(f"n_img shape {n_image.shape}")
    return n_image
#def behaviorAnalysis()

p_center = [0, 0]  

boxCoords = []
boxCoords2 = []
boxCenter = []
tempW = []
tempH = []
m_circle = None
m_centers = None
j = 0
collection = []
currentBehavior = None
positionBuffer = []
frameBuffer = []

count = 0
while(1):
    count += 1
    ret, frame = cap.read()
    # if (j < 50):
    #     j+=1
    #     continue
    cframe = np.copy(frame)
    c_cleanFrame = np.copy(frame)
    nFrame = np.copy(frame)
    
    height,width, c_channels = frame.shape

    # Background Subtractor
    mask = np.zeros((height,width, c_channels), np.uint8)
    frame1 = cv2.GaussianBlur(frame, (5,5), cv2.BORDER_DEFAULT)
    fgmask = fgbg.apply(frame1)
    
    #edges = cv2.Canny(frame, 200, 100)
    
    ret,thresh = cv2.threshold(fgmask,127,255,0)
    
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    c_mask = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2RGB) 
    maskedImg = cv2.bitwise_and(frame1, mask)
    
    if len(contours) != 0:
        # draw in blue the contours that were founded`````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        cv2.drawContours(maskedImg, c, -1, 255, 3)
    
        x,y,w,h = cv2.boundingRect(c)

        c_center = center(x,y,w,h)
              
        #print(f"Current Center: {c_center[0]} {c_center[1]}    Previous Center: {p_center[0]} {p_center[1]}")

        #print(f"Distance: {distance(c_center, p_center)}")
        
        p_center = c_center

        if w > MAX_LENGTH or h > MAX_LENGTH or w < MIN_LENGTH or h < MIN_LENGTH:
            if len(boxCoords):
                cv2.rectangle(frame,(boxCoords[-1][0],boxCoords[-1][1]),(boxCoords[-1][0]+boxCoords[-1][2],boxCoords[-1][1]+boxCoords[-1][3]),(0,255,0),2)
        else:
            # draw the biggest contour (c) in green
            if len(boxCoords) > 0:
                prevCenter = center(boxCoords[-1][0],boxCoords[-1][1],boxCoords[-1][2],boxCoords[-1][3])
                m_distance = distance(c_center, prevCenter)
                if m_distance < DISTANCE_THRESHOLD:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    boxCoords.append([x,y,w,h])
                    tempH.append(h)
                    tempW.append(w)
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                boxCoords.append([x,y,w,h])
                tempH.append(h)
                tempW.append(w)
                
            print(f"x:{x} y:{y} w:{w} h:{h}")

    cv2.imshow("mask", maskedImg)

    if len(boxCoords):
        print(f"x:{boxCoords[-1][0]} y:{boxCoords[-1][1]} w:{boxCoords[-1][2]} h:{boxCoords[-1][3]}")
        boxFrame = frame[boxCoords[-1][1]:boxCoords[-1][1] + boxCoords[-1][3], boxCoords[-1][0]:boxCoords[-1][0] + boxCoords[-1][2]]         
        #cv2.imshow('boxFrame',boxFrame)
    
    # END BACKGROUND SUBTRACTOR

    # Hough Circle Transform
    if m_circle is None:
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        edges = cv2.Canny(gray, 100, 200)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10000, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)
        for i in circles:
            m_circle = i[0]    

    i = m_circle
    circle_img = cv2.circle(c_mask,(int(i[0]),int(i[1])),int(i[2]),(255,255,255),thickness=-1)

    masked_data = cv2.bitwise_and(frame1, circle_img)   
    masked_data[np.where((masked_data==[0,0,0]).all(axis=2))] = [255,255,255]  

    cv2.imshow("maskedData", masked_data)

    t_maskImg = np.copy(masked_data)
    fgmask = fgbg2.apply(t_maskImg)
    cv2.imshow("temp", fgmask)


    gray = cv2.cvtColor(masked_data, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (20,20))
    
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(thresh, 550, 450)

    contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)

    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])

    cY = int(M["m01"] / M["m00"])

    cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

    cv2.drawContours(frame, c, -1, 255, 3)

    if (cv2.contourArea(c) > 200):
        c_mask = np.zeros((height,width,c_channels), np.uint8)
        bkgd = np.copy(c_mask)
        cv2.drawContours(bkgd, c, -1, (0,255,0), 3)
        x,y,w,h = cv2.boundingRect(c)
        
        cropped = bkgd[y:y+h, x:x+w]
        padded = imagePadder(cropped, (200, 200))
        cv2.imshow("padded", padded)
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        cv2.imshow("grayPadded", gray)

        cv2.fillPoly(c_mask, pts =[c], color=(255,255,255))
        frame4 = np.copy(nFrame)
        frame4[np.where((c_mask==[0,0,0]).all(axis=2))] = [255,255,255]
        n_x, n_y, n_w, n_h = offsetAdder(x, y, w , h, OFFSET, width, height)
        print(f"{h} {n_h}")
        nCropped = nFrame[n_y:n_y+n_h, n_x:n_x+n_w]
        nPadded = imagePadder(nCropped, (200, 200))
        earDetect = earDetection(nCropped, detect_fn)
        nFrame[n_y:n_y+n_h, n_x:n_x+n_w] = earDetect
        i = np.copy(nPadded)
        if i.shape[0] == 199:
            i = cv2.copyMakeBorder(i, 0, 1, 0, 0, cv2.BORDER_CONSTANT)
 
        if i.shape[1] == 199:
            i = cv2.copyMakeBorder(i, 0, 0, 1, 0, cv2.BORDER_CONSTANT)
           
        if i.shape[0] == 199:
            i = cv2.copyMakeBorder(i, 1, 0, 0, 0, cv2.BORDER_CONSTANT)
           
        if i.shape[1] == 199:
            i = cv2.copyMakeBorder(i, 0, 0, 0, 1, cv2.BORDER_CONSTANT)
            
        
        frameBuffer.append(i)

        if len(frameBuffer) > 5:
            frameBuffer.pop(0)

        positionBuffer.append([cX, cY])
        if len(positionBuffer) > 15:
            positionBuffer.pop(0)

        if len(frameBuffer) == 5 and len(positionBuffer) == 15:
            np_img1 = np.array(frameBuffer)
            np_centers = np.array(positionBuffer)
            tf_img1 = tf.convert_to_tensor(np_img1)
            tf_centers = tf.convert_to_tensor(np_centers)
            detectedBehavior = behaviorAnalysis(tf_img1[0], tf_img1[2], tf_img1[4], tf_centers)
            print(f"Detected Behavior {detectedBehavior}")
            cv2.putText(frame, detectedBehavior, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # if count % 4 == 0:
        #     cv2.imwrite("V:/Users/Shrayes/Dask/Recordings/imgs/vid3/frame%d.jpg" % count, nPadded)

        cv2.imshow("frame4", nFrame)
        if len(collection) == 2:
            collection.pop(0)
        collection.append(frame4)
        
    # if len(collection) == 2:
    #     img = np.array(collection)
    #     print(img.shape)
    #     img = [createMotionImage(img)]
    #     cv2.imshow("heatmap", img[0])
    #     img = np.array(img)
    #     print(img.shape)
    #     # img = tf.convert_to_tensor(img)
    #     # img = tf.image.resize(img, [int(height/4), int(width/4)])
    #     # predictions = model.predict(img)
    #     # prediction = m_labels[np.argmax(predictions[0])]
    #     # currentbehavior = prediction
    #     # print(currentBehavior)
        
        
    flag = False
    for i in contours:
        if cv2.contourArea(i) > 500:
            x,y,w,h = cv2.boundingRect(i)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            cv2.imshow("hello2", c_cleanFrame[y:y+h, x:x+w])
            boxCoords2.append([x,y,w,h])
            flag = True
            if currentBehavior is not None:
                print("hello")
                cv2.putText(c_cleanFrame, currentBehavior, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    if flag == False:
        if len(boxCoords2):
                cv2.rectangle(frame,(boxCoords2[-1][0],boxCoords2[-1][1]),(boxCoords2[-1][0]+boxCoords2[-1][2],boxCoords2[-1][1]+boxCoords2[-1][3]),(0,255,0),2)
    
    # K-means
    if m_centers is None:
        if len(boxCoords2):
            boxFrame = frame[boxCoords2[-1][1]:boxCoords2[-1][1] + boxCoords2[-1][3], boxCoords2[-1][0]:boxCoords2[-1][0] + boxCoords2[-1][2]] 
            flattened_frame = boxFrame.reshape((-1,3))
            flattened_frame = np.float32(flattened_frame)
            n_flat_frame = np.delete(flattened_frame, np.where(flattened_frame==[0,0,0]))
            print(n_flat_frame.shape)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

            k = 2
            _, labels, (centers) = cv2.kmeans(flattened_frame, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # convert back to 8 bit values
            centers = np.uint8(centers)
            m_centers = centers

            # flatten the labels array
            labels = labels.flatten()
            segmented_image = centers[labels.flatten()]
            segmented_image = segmented_image.reshape(boxFrame.shape)

    cv2.imshow('frame', frame)

    # show one frame at a time
    key = cv2.waitKey(30)
    # Quit when 'q' is pressed
    if key == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
