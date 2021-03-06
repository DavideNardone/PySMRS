from SMRS import SMRS

import cv2
from matplotlib import pyplot as plt
import os
import sys
import numpy as np
import time


def saveImages(Y, repInd, dir_output):

    if os.path.exists(dir_output)==False:
        os.mkdir(dir_output)

    repInd.sort()

    print ('Saving images...')       
    for i in repInd:
        plt.imshow(np.reshape(Y[:,i],(240,320)), 'gray')
        plt.axis('off')
        fig.savefig(dir_output + 'frame_' + str(i) +'.png')

def plot_sparsness(C):

    plt.spy(C)
    plt.show()


if __name__ == '__main__':

    dir_video = 'dataset/Society Raffles.mp4'
    dir_output = 'output/'

    if os.path.isfile(dir_video) or os.access(dir_video, os.R_OK):
        print ('Reading video...')
    else:
        print "The file is missing or not readable!"
        sys.exit()

    vidcap = cv2.VideoCapture(dir_video)

    success, image = vidcap.read()
    count = 0
    success = True
    
    Y = []
    #TODO: Implement video sub-sample
    print ('Reading video...')
    while success:
        success, frame = vidcap.read()

        if frame is None:
            print ('Done reading')
        else:
            #for convention we convert RGB into a gray scale format
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            #stacking frames into a matrix
            Y.append(gray_frame.flatten())
    
    # setting frames on the column and pixels on the rows
    Y = np.transpose(np.asarray(Y))

    print ('Problem size: [%d,%d]' % (Y.shape[0],Y.shape[1]))
    print ('Extracting the representatives frames from the video...It may takes a while...')
    start_time = time.time()
    smrs = SMRS(data=Y,
                alpha=5,
                norm_type=1,
                verbose=True, thr=[10**-8],
                thrS=0.99,
                thrP=0.50,
                max_iter=5000,
                affine=True,
                normalize=False,
                step=1,
                PCA=False,
                GPU=False)

    sInd, repInd, C = smrs.smrs()
    
    saveImages(Y,repInd,dir_output)

    plot_sparsness(C)


