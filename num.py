from cv2 import cv2
import numpy as np
import os

digits = [1,2,3,4,5,6,7,8,9,0]
ndic = []

d = os.listdir()

for f in d:
    for digit in digits:
        if f == f"{str(digit)}.npy":
            ndic.append(np.load(f))

for c,e in enumerate(ndic):
    cv2.imshow(str(c), e)

 # show one frame at a time
key = cv2.waitKey(0)

cv2.destroyAllWindows()
