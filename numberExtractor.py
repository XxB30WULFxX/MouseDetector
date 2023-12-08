import cv2 
import numpy as np
import os

#cap = cv2.VideoCapture("video_20211030_143455.h264")

class NumberExtractor:
    def __init__(self, pathToDigits):
        self.digits = [0,1,2,3,4,5,6,7,8,9,"-","A", "period", "colon", "onepoint"]
        self.ndic = []

        d = os.listdir(pathToDigits)
    
        for digit in self.digits:
            for f in d:
                if f == f"{str(digit)}.npy":
                    print(f"Loading {digit}")
                    self.ndic.append(np.load(os.path.join(pathToDigits,f)))

    def extract(self, frame):

        chars = self.frameMask(frame)

        nums = []

        for char in chars:
            arr = []
            for i, digit in enumerate(self.ndic):
                c = self.nrectFrame[:,char[0]:char[1],:]
                corr = cv2.matchTemplate(c, digit, 
                            cv2.TM_CCOEFF_NORMED)[0][0]
                arr.append(corr)
            nums.append(arr.index(max(arr)))

        number = ""
        i = 0
        for c in nums:
            if i == 10 or i == 23:
                number += " "
            if self.digits[c] == "colon":
                number += ":"
            elif self.digits[c] == "period":
                number += "."
            elif self.digits[c] == "onepoint":
                number += "1"
            else:
                number += str(self.digits[c])
            i += 1
            
        return number

    def frameMask(self, frame):
        rectFrame = frame[20:50, 380:-380, :] 
        self.nrectFrame = rectFrame.copy()

        c = False
        wPixel = 0

        chars = []

        lastColumn = None

        for column in range(rectFrame.shape[1]):
            white = False
            for row in range(rectFrame.shape[0]):
                if all(rectFrame[row][column] > [230,230,230]):
                    white=True

                    if c:
                        wPixel += 1

            if white==True and c==False:       
                lastColumn = column
                c = True
                cv2.line(rectFrame,(column,0),(column, rectFrame.shape[0] - 1),(0,0,255),1)

            elif white==False and c==True:
                #print(wPixel)
                wPixel = 0
                c = False
                chars.append((lastColumn, column))
                cv2.line(rectFrame,(column,0),(column, rectFrame.shape[0] - 1),(0,255,0),1)

        return chars

"""


counter = 0

while True: 
    print(f"counter:{counter} ")
    counter += 1
    ret, frame = cap.read()

    if not ret:
        break


    rectFrame = frame[20:50, 380:-380, :] 
    nrectFrame = rectFrame.copy()

    c = False
    wPixel = 0

    chars = []

    lastColumn = None

    for column in range(rectFrame.shape[1]):
        white = False
        for row in range(rectFrame.shape[0]):
            if all(rectFrame[row][column] > [230,230,230]):
                white=True

                if c:
                    wPixel += 1

        if white==True and c==False:       
            lastColumn = column
            c = True
            cv2.line(rectFrame,(column,0),(column, rectFrame.shape[0] - 1),(0,0,255),1)

        elif white==False and c==True:
            #print(wPixel)
            wPixel = 0
            c = False
            chars.append((lastColumn, column))
            cv2.line(rectFrame,(column,0),(column, rectFrame.shape[0] - 1),(0,255,0),1)

    cv2.imshow("frame", nrectFrame)

    # # show one frame at a time
    # key = cv2.waitKey(0)

    # kinput = input("This frame? ")

    # if kinput == "n":
    #     cv2.destroyAllWindows()
    #     continue

    # for c, i in enumerate(chars):
        
    #     cv2.imshow(str(c), nrectFrame[:,i[0]:i[1],:])
    #     # show one frame at a time
    #     nkey = cv2.waitKey(0)
    #     print("h")
    #     inp = input("Enter ")
    #     if inp == "p":
    #         print("skipping!")
    #         cv2.destroyAllWindows()
    #         continue
    #     np.save(f"{inp}.npy", nrectFrame[:,i[0]:i[1],:])
    #     cv2.destroyAllWindows()

    digits = [0,1,2,3,4,5,6,7,8,9,"-","A", "period", "colon", "onepoint"]
    ndic = []

    d = os.listdir()
    for digit in digits:
        for f in d:
            if f == f"{str(digit)}.npy":
                print(f"Loading {digit}")
                t = np.load(f)
                
                ndic.append(t)        

    nums = []
    print("====================")
    for char in chars:
        arr = []
        for i, digit in enumerate(ndic):
            c = nrectFrame[:,char[0]:char[1],:]

            print(c.shape)
            print(digit.shape)
        
            corr = cv2.matchTemplate(c, digit, 
                        cv2.TM_CCOEFF_NORMED)[0][0]
            arr.append(corr)
        nums.append(arr.index(max(arr)))

    number = ""
    i = 0
    for c in nums:
        print(f"digits: {digits[c]}")
        if i == 10 or i == 23:
            number += " "
        if digits[c] == "colon":
            number += ":"
        elif digits[c] == "period":
            number += "."
        elif digits[c] == "onepoint":
            number += "1"
        else:
            number += str(digits[c])
        i += 1
        
    print(number)
    

    key = cv2.waitKey(0)

    cv2.destroyAllWindows()

    #Quit when 'q' is pressed
    if key == ord('q'):
        break



cap.release()
"""