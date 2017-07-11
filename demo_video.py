from SMRS import SMRS

import cv2
import os
import sys
import numpy as np


if __name__ == '__main__':

    vidcap = cv2.VideoCapture('/Users/davidenardone/Desktop/Society Raffles.mp4')

    success, image = vidcap.read()
    count = 0
    success = True
    Y = []
    print ('Reading video...')
    while success:
        success, frame = vidcap.read()

        if frame is None:
            print ('None')
        else:
            #for convention we convert RGB into a gray scale format
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            #stacking frames into a matrix
            Y.append(gray_frame.flatten())

    Y = np.asarray(Y)

    print ('Extracting the representatives frames from the video...It may takes a while...')
    smrs = SMRS(data=Y, alpha=5, dim_red=0,norm_type=2,
                thr=10**-8, max_iter=5000, affine=True,
                verbose=False)

    rep_ind, C = smrs.smrs()