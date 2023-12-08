import numpy as np
from cv2 import cv2
from math import sqrt

cap = cv2.VideoCapture('video_20210101_133717.h264')

MAX_LENGTH = 250
MIN_LENGTH = 50
DISTANCE_THRESHOLD = 0

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG() 

def distance(p1, p2):
    # Return distance between two points
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def center(x, y, w, h):
    # Return center of box
    center = []
    center.append(x + (w / 2))
    center.append(y - (h / 2))
    return center

p_center = [0, 0]  

boxCoords = []
boxCenter = []
tempW = []
tempH = []

while(1):
    ret, frame = cap.read()

    frame = frame[:912,:,:]

    fgmask = fgbg.apply(frame)
    

    
    ret,thresh = cv2.threshold(fgmask,127,255,0)
    
    img, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))
    mask = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2RGB) 
    maskedImg = cv2.bitwise_and(frame, mask)

    
    
    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(maskedImg, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

        c_center = center(x,y,w,h)
              
        # print(f"Current Center: {c_center[0]} {c_center[1]}    Previous Center: {p_center[0]} {p_center[1]}")

        # print(f"Distance: {distance(c_center, p_center)}")
        
        p_center = c_center

        if w > MAX_LENGTH or h > MAX_LENGTH or w < MIN_LENGTH or h < MIN_LENGTH:
            pass
        else:
            # draw the biggest contour (c) in green
            cv2.rectangle(maskedImg,(x,y),(x+w,y+h),(0,255,0),2)
            boxCoords.append([x,y,w,h])
            tempH.append(h)
            tempW.append(w)


    cv2.imshow('frame2',fgmask)
    cv2.imshow('frame',maskedImg)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
print(f"Minimum width: {min(tempW)} height: {min(tempH)}")

cap.release()
cv2.destroyAllWindows()

