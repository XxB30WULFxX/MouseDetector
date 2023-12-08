import cv2 
import numpy as np
import os
from numberExtractor import NumberExtractor as ne

numExtractor = ne()

TIME_STAMP = 5


inputSource = "video_20211023_133202.mp4"
cap = cv2.VideoCapture(inputSource)
fps = cap.get(cv2.CAP_PROP_FPS)

ret = True

count = 0

times = []

while(ret):
    ret, frame = cap.read()
    if TIME_STAMP:
        if count % (TIME_STAMP*60*fps) == 0:
            print(f"Processing Min {count / (60*fps)}")

    count += 1

    if count % fps != 0:
        continue

    t = numExtractor.extract(frame)
    times.append([count, t[11:16]])

cap.release()
cv2.destroyAllWindows()


