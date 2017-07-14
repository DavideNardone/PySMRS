from SMRS import SMRS

import cv2
from matplotlib import pyplot as plt
import os
import sys
import numpy as np
import time



def subPlotsRepInd(Y, rep_ind, n_rows=2):

    n_cols = int(np.ceil(rep_ind.shape[0]/n_rows))
    #TODO: set title image
    for i in xrange(0,len(rep_ind)):
        _i = int(str(n_rows)+str(n_cols) + str(i+1))
        plt.subplot(_i), plt.imshow(np.reshape(Y[i],(240,320)), 'gray')
        plt.xticks([])
        plt.yticks([])


    plt.show()


def plot_sparsness(C):

    plt.spy(C)
    plt.show()


if __name__ == '__main__':

    vidcap = cv2.VideoCapture('/Users/davidenardone/PycharmProjects/SMRS/dataset/Society Raffles.mp4')

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
            # print (gray_frame.shape)
            # sys.exit()
            # cv2.imshow('image', gray_frame)
            # cv2.waitKey(0)
            # plt.show()
            # sys.exit()

            #stacking frames into a matrix
            Y.append(gray_frame.flatten())

    Y = np.asarray(Y)

    print ('Problem size: [%d,%d]' % (Y.shape[0],Y.shape[1]))
    print ('Extracting the representatives frames from the video...It may takes a while...')
    start_time = time.time()
    smrs = SMRS(data=Y, alpha=2, norm_type=2,
                verbose=True, thr=10**-7, max_iter=5000,
                affine=True,
                PCA=False, GPU=False)

    rep_ind, C = smrs.smrs()

    subPlotsRepInd(Y,rep_ind)

    plot_sparsness(C)


