from __future__ import division

from sklearn.decomposition import PCA
from scipy import linalg

import numpy as np
import numpy.matlib
import os
import sys
import hdf5storage


class SMRS():

    def __init__(self, data, alpha=10, dim_red=0,norm_type=1,
                verbose=False, thr=1*10^-7, max_iter=5000,
                affine=False):

        self.data = data
        self.alpha = alpha
        self.dim_red = dim_red
        self.norm_type=norm_type
        self.verbose = verbose
        self.thr = thr
        self.max_iter = max_iter
        self.affine = affine

        self.num_rows = data.shape[0]
        self.num_columns = data.shape[1]

    def computeLambda(self):

        if not self.affine:
            T = np.zeros(self.num_columns)
            for i in xrange(0,self.num_columns):
                yi = self.data[:,i]
                T[i] = np.linalg.norm(np.dot(yi.T, self.data))
        else:
            T = np.zeros(self.num_columns)

            for i in xrange(0, self.num_columns):
                yi = self.data[:, i]
                y_mean = np.mean(self.data,axis=1)

                # norm(yi' * (ymean*ones(1,N)-Y));
                T[i] = np.linalg.norm(np.dot(yi.T, np.outer(y_mean, np.ones(self.num_columns)) - self.data))

        _lamda = np.amax(T)

        return _lamda

    def shrinkL1Lq(self, C1, _lambda):

        D,N = C1.shape

        if self.norm_type == 1:

            #TODO: incapsulate into one function
            # soft thresholding
            C2 = np.abs(C1) - _lambda
            ind = C2 < 0
            C2[ind] = 0
            C2 = np.multiply(C2, np.sign(C1))
        elif self.norm_type == 2:
            r = np.zeros([D,1])
            for j in xrange(0,D):
                th = np.linalg.norm(C1[j,:]) - _lambda
                r[j] = 0 if th < 0 else th
            C2 = np.multiply(np.matlib.repmat(np.divide(r, (r + _lambda )), 1, N), C1)
            print (C2)
        elif self.norm_type == 'inf':
            # TODO: write it
            print ''



        # elif self.norm_type == 2:
        #     print ''
        # elif self.norm_type == inf:
        #     print ''

        return C2


    def almLasso_mat_fun(self):
        print ('ADMM processing...')

        alpha1 = alpha2 = 0
        if (len(self.reg_params) == 1):
            alpha1 = self.reg_params[0]
            alpha2 = self.reg_params[0]
        elif (len(self.reg_params) == 2):
            alpha1 = self.reg_params[0]
            alpha2 = self.reg_params[1]

        thr1 = self.thr
        thr2 = self.thr

        mu1p = alpha1 * 1/self.computeLambda()
        mu2p = alpha2 * 1

        mu1 = mu1p
        mu2 = mu2p
        P = self.data.T.dot(self.data)
        A = linalg.inv(np.multiply(mu1,P) +  np.multiply(mu2, np.eye(self.num_columns, dtype=int)) +  np.multiply(mu2, np.ones([self.num_columns,self.num_columns]) ))

        C1 = np.zeros([self.num_columns,self.num_columns])
        lambda2 = np.zeros([self.num_columns, self.num_columns])
        lambda3 = np.zeros(self.num_columns).T
        err1 = 10*thr1
        err2 = 10*thr2
        i = 1

        while ( (err1 > thr1 | err2 > thr1) & i < self.max_iter):

            OP1 = np.multiply(P,mu1)
            OP2 = np.multiply(C1 - np.divide(lambda2,mu2), mu2)
            OP3 = np.multiply(mu2, np.ones([self.num_columns,self.num_columns]))
            OP4 =np.matlib.repmat(lambda3, self.num_columns, 1)
            Z = A.dot(OP1 + OP2 + OP3 + OP4 )

            C1 = Z + np.divide(lambda2,mu2)
            # print (C1)
            _lambda = 1/mu2
            C2 = self.shrinkL1Lq(C1, _lambda)

        # C1 = zeros(N,N);
        # Lambda2 = zeros(N,N);
        # lambda3 = zeros(1,N);
        # err1 = 10*thr1; err2 = 10*thr2;
        # i = 1;




    def smrs(self):
        self.reg_params = [self.alpha, self.alpha]

        self.thrS = 0.99
        self.thP = 0.95

        #data normalization

        # print(np.mean(self.data, axis=1))
        self.data = self.data - np.matlib.repmat(np.mean(self.data, axis=1), self.num_columns,1).T
        # print (self.data)

        if (self.dim_red != 0):
            n_comp = params['n_components']
            pca = PCA(n_components = self.dim_red)
            self.data = pca.fit_transform(self.data)



        C = self.almLasso_mat_fun()


        return 0,C

if __name__ == '__main__':

    mc = hdf5storage.loadmat('/home/davidenardone/PySMRS/data.mat')
    data = mc['data']


    smrs = SMRS(data=data, alpha=5, dim_red=0,norm_type=2,
                verbose=False, thr=1*10^-7, max_iter=5000,
                affine=True)


    rep_ind, C = smrs.smrs()


