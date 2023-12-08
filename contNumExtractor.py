import cv2 
import numpy as np
import os
from numberExtractor import NumberExtractor as ne
import pandas as pd

class contNumExtractor:
    def __init__(self, time_stamp=5) -> None:

        self.numExtractor = ne()

        self.TIME_STAMP = time_stamp

    def run(self, inputSource, outputFolder=None):

        n_inputSource = inputSource.split('/')[-1]
        print(inputSource)
        print(n_inputSource)

        cap = cv2.VideoCapture(inputSource)
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(fps)

        ret = True

        count = 0

        times = []

        while(ret):
            ret, frame = cap.read()
            if self.TIME_STAMP:
                if count % (self.TIME_STAMP*60*fps) == 0:
                    print(f"Processing Min {count / (60*fps)}")

            count += 1

            if count % fps != 0:
                continue

            t = self.numExtractor.extract(frame)
            times.append([count, t])

        df = pd.DataFrame(times, columns=["frame_num", "time_stamp"])

        if outputFolder:
            df.to_hdf(f"{os.path.join(outputFolder, n_inputSource)}.h5", key="data")
        else:
            df.to_hdf(f"{n_inputSource}.h5", key="data")

        cap.release()
        cv2.destroyAllWindows()


