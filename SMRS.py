from __future__ import division

from sklearn.decomposition import PCA

import numpy as np
import numpy.matlib
np.set_printoptions(threshold=np.inf)
import numpy.matlib
import sys
import hdf5storage
import scipy.io
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

import skcuda.linalg as linalg
import skcuda.misc as misc
import time



class SMRS():

    def __init__(self, data, alpha=10, norm_type=1,
                verbose=False, thr=10**-8, max_iter=5000,
                affine=False,
                PCA=False, npc=10, GPU=False):

        self.data = data
        self.alpha = alpha
        self.norm_type=norm_type
        self.verbose = verbose
        self.thr = thr
        self.max_iter = max_iter
        self.affine = affine
        self.PCA = PCA
        self.npc = npc
        self.GPU = GPU

        self.num_rows = data.shape[0]
        self.num_columns = data.shape[1]

        if(self.GPU==True):
            linalg.init()

            print (misc.get_current_device())


    def computeLambda(self):
        print ('Computing lambda...')

        _lambda = []
        T = np.zeros(self.num_columns)

        if (self.GPU == True):


            if not self.affine:

                gpu_data = gpuarray.to_gpu(self.data)


                C_gpu = linalg.dot(gpu_data, gpu_data, transa='T')

                for i in xrange(self.num_columns):
                    T[i] = linalg.norm(C_gpu[:,i])
            else:

                gpu_data = gpuarray.to_gpu(self.data)

                # affine transformation
                y_mean_gpu = misc.mean(gpu_data,axis=1)

                # y_mean = np.mean(self.data,axis=1)

                # creating affine matrix to subtract to the data (may encounter problem with strides)
                aff_mat = np.zeros([self.num_rows,self.num_columns]).astype('f')
                for i in xrange(0,self.num_columns):
                    aff_mat[:,i] = y_mean_gpu.get()


                aff_mat_gpu = gpuarray.to_gpu(aff_mat)
                gpu_data_aff = misc.subtract(aff_mat_gpu,gpu_data)


                C_gpu = linalg.dot(gpu_data, gpu_data_aff, transa='T')

                #computing euclidean norm
                for i in xrange(self.num_columns):
                    T[i] = linalg.norm(C_gpu[:,i])

        else:

            if not self.affine:

                T = np.linalg.norm(np.dot(self.data.T, self.data), axis=1)

            else:
                #affine transformation
                y_mean = np.mean(self.data, axis=1)

                tmp_mat = np.outer(y_mean, np.ones(self.num_columns)) - self.data

                T = np.linalg.norm(np.dot(self.data.T, tmp_mat),axis=1)

        _lambda = np.amax(T)

        print (_lambda)
        return _lambda


    def shrinkL1Lq(self, C1, _lambda):

        D,N = C1.shape
        C2 = []
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
        elif self.norm_type == 'inf':
            # TODO: write it
            print ''
        # elif self.norm_type == 2:
        #     print ''
        # elif self.norm_type == inf:
        #     print ''

        return C2








    def errorCoef(self, Z, C):

        err = np.sum(np.abs(Z-C)) / (np.shape(C)[0] * np.shape(C)[1])

        return err
        # err = sum(sum(abs(Z - C))) / (size(C, 1) * size(C, 2));


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

        start_time = time.time()
        mu1p = alpha1 * 1/self.computeLambda()
        print("-Compute Lambda- Time = %s seconds" % (time.time() - start_time))
        mu2p = alpha2 * 1

        mu1 = mu1p
        mu2 = mu2p

        err1 = 10 * thr1
        err2 = 10 * thr2

        i = 1
        Err = []
        C2 = []

        start_time = time.time()
        if self.GPU == True:

            linalg.init()

            gpu_data = gpuarray.to_gpu(self.data)
            P_GPU = linalg.dot(gpu_data,gpu_data,transa='T')

            OP1 = P_GPU
            linalg.scale(mu1, OP1)

            OP2 = linalg.eye(self.num_columns)
            linalg.scale(mu2,OP2)

            C1 = misc.zeros((self.num_columns,self.num_columns),dtype='float32')
            lambda2 = misc.zeros((self.num_columns,self.num_columns),dtype='float32')

            if self.affine == True:

                OP3 = misc.ones((self.num_columns, self.num_columns), dtype='float32')
                linalg.scale(mu2, OP3)
                lambda3 = misc.zeros((1, self.num_columns), dtype='float32')


                # TODO: Because of some problem with linalg.inv version of scikit-cuda we fix it using np.linalg.inv of numpy
                A = np.linalg.inv(misc.add(misc.add(OP1.get(), OP2.get()), OP3.get()))
                A_GPU = gpuarray.to_gpu(A)

                #GPU version may not converge ...
                while ( (err1 > thr1 or err2 > thr1) and i < self.max_iter):

                    _lambda2 = lambda2.astype('float32')
                    _lambda3 = lambda3.astype('float32')

                    linalg.scale(1.0/mu2, _lambda2)
                    term_OP2 = gpuarray.to_gpu(_lambda2.get())

                    OP2 = misc.subtract(C1, term_OP2)
                    OP4 = gpuarray.to_gpu(np.matlib.repmat(_lambda3.get(), self.num_columns, 1))

                    # updating Z
                    Z = linalg.dot(A_GPU,misc.add(misc.add(misc.add(OP1,OP2),OP3),OP4))

                    # updating C
                    C1 = misc.add(Z,term_OP2)
                    C2 = self.shrinkL1Lq(C1.get(),1/mu2)
                    C2 = C2.astype('float32')

                    # updating Lagrange multipliers
                    term_lambda2 = misc.subtract(Z, gpuarray.to_gpu(C2))
                    linalg.scale(mu2,term_lambda2)
                    term_lambda2 = gpuarray.to_gpu(term_lambda2.get())
                    lambda2 = misc.add(lambda2, term_lambda2) # on GPU

                    term_lambda3 = misc.subtract(misc.ones((1, self.num_columns), dtype='float32'), misc.sum(Z,axis=0))
                    linalg.scale(mu2,term_lambda3)
                    term_lambda3 = gpuarray.to_gpu(term_lambda3.get())
                    lambda3 = misc.add(lambda3, term_lambda3) # on GPU

                    err1 = self.errorCoef(Z.get(), C2)
                    err2 = self.errorCoef(np.sum(Z.get(), axis=0), np.ones([1, self.num_columns]))

                    # mu1 = min(mu1 * (1 + 10 ^ -5), 10 ^ 2 * mu1p);
                    # mu2 = min(mu2 * (1 + 10 ^ -5), 10 ^ 2 * mu2p);

                    C1 = gpuarray.to_gpu((C2))

                    i += 1
                    # reporting errors
                    # if (self.verbose &  (i % 5 == 0)):
                    print('Iteration = %d, ||Z - C|| = %2.5e, ||1 - C^T 1|| = %2.5e' % (i, err1, err2))

                Err = [err1, err2]
                if (self.verbose):
                    print ('Terminating ADMM at iteration %5.0f, \n ||Z - C|| = %2.5e, ||1 - C^T 1|| = %2.5e. \n' % (i, err1, err2))
            else:
                print 'GPU not affine'

                # TO TEST !!!!!!!!!!!!!!!!

                # TODO: Because of some problem with linalg.inv version of scikit-cuda we fix it using np.linalg.inv of numpy
                A = np.linalg.inv(misc.add(OP1.get(), OP2.get()))
                A_GPU = gpuarray.to_gpu(A)

                while ( err1 > thr1 and i < self.max_iter):

                    _lambda2 = lambda2

                    term_OP2 = misc.subtract(C1, _lambda2)

                    linalg.scale(mu2, term_OP2)
                    OP2 = gpuarray.to_gpu(term_OP2.get())

                    Z = linalg.dot(A_GPU, misc.add(OP1 , OP2))
                    print (Z)
                    sys.exit()

                    # Z = A * (mu1.*P+mu2.*C1-Lambda2);

        else: #CPU version

            if self.affine == True:

                P = self.data.T.dot(self.data)
                A = np.linalg.inv(np.multiply(mu1,P) +  np.multiply(mu2, np.eye(self.num_columns, dtype=int)) +  np.multiply(mu2, np.ones([self.num_columns,self.num_columns]) ))

                C1 = np.zeros([self.num_columns,self.num_columns])
                lambda2 = np.zeros([self.num_columns, self.num_columns])
                lambda3 = np.zeros(self.num_columns).T

                OP1 = np.multiply(P, mu1)
                OP3 = np.multiply(mu2, np.ones([self.num_columns, self.num_columns]))


                while ( (err1 > thr1 or err2 > thr1) and i < self.max_iter):


                    OP2 = np.multiply(C1 - np.divide(lambda2,mu2), mu2)
                    OP4 = np.matlib.repmat(lambda3, self.num_columns, 1)

                    # updating Z
                    Z = A.dot(OP1 + OP2 + OP3 + OP4 )

                    # updating C
                    C1 = Z + np.divide(lambda2,mu2)
                    C2 = self.shrinkL1Lq(C1, 1/mu2)

                    # updating Lagrange multipliers
                    lambda2 = lambda2 + np.multiply(mu2,Z - C2)
                    lambda3 = lambda3 + np.multiply(mu2, np.ones([1,self.num_columns]) - np.sum(Z,axis=0))

                    err1 = self.errorCoef(Z, C2)
                    err2 = self.errorCoef(np.sum(Z,axis=0), np.ones([1, self.num_columns]))

                    # mu1 = min(mu1 * (1 + 10 ^ -5), 10 ^ 2 * mu1p);
                    # mu2 = min(mu2 * (1 + 10 ^ -5), 10 ^ 2 * mu2p);

                    C1 = C2
                    i += 1
                    # reporting errors
                    # if (self.verbose &  (i % 5 == 0)):
                    print('Iteration = %d, ||Z - C|| = %2.5e, ||1 - C^T 1|| = %2.5e' % (i, err1, err2))

                Err = [err1, err2]
                if (self.verbose):
                    print ('Terminating ADMM at iteration %5.0f, \n ||Z - C|| = %2.5e, ||1 - C^T 1|| = %2.5e. \n' % (i, err1,err2))
                else:
                    print 'CPU not affine'


        print("-ADMM- Time = %s seconds" % (time.time() - start_time))
        return C2, Err

    def rmRep(self, sInd, thr):

        Ys = self.data[:, sInd]

        Ns = Ys.shape[1]
        d = np.zeros([Ns, Ns])

        for i in xrange(0,Ns-1):
            for j in xrange(i+1,Ns):
                d[i,j] = np.linalg.norm(Ys[:,i] - Ys[:,j])

        d = d + d.T

        dsorti = np.argsort(d,axis=0)[::-1]
        dsort = np.flipud(np.sort(d,axis=0))

        pind = np.arange(0,Ns)
        t = []
        for i in xrange(0, Ns):
            if len(np.where(pind==i)):
                cum = 0
                t = -1
                while cum <= (thr * np.sum(dsort[:,i])):
                    t += 1
                    cum += dsort[t, i]

                pind = np.setdiff1d(pind, np.setdiff1d( dsorti[t:,i], np.arange(0,i+1), assume_unique=True), assume_unique=True)

        ind = sInd[pind]

        return ind



    def findRep(self,C, thr, norm):
        print ('Finding most representative objects')

        N = C.shape[0]

        r = np.zeros([1,N])

        for i in xrange(0, N):

            r[0,i] = np.linalg.norm(C[i,:],  norm)
            # print (np.linalg.norm(C[i,:]))
        # print (r.shape)
        nrmInd = np.argsort(r)[0][::-1] #descending order
        # print (nrmInd)
        nrm = r[0,nrmInd]
        # print (nrm)
        nrmSum = 0


        j=[]
        for j in xrange(0,N):
            nrmSum = nrmSum + nrm[j]
            if ((nrmSum/np.sum(nrm)) > thr):
                break

        cssInd = nrmInd[0:j+1]

        return cssInd


    def smrs(self):
        self.reg_params = [self.alpha, self.alpha]

        thrS = 0.99
        thrP = 0.95

        #data normalization
        self.data = self.data - np.matlib.repmat(np.mean(self.data, axis=1), self.num_columns,1).T


        if (self.PCA == True):
            print ('Performing PCA...')
            pca = PCA(n_components = self.npc)
            self.data = pca.fit_transform(self.data)
            self.num_columns = self.data.shape[0]
            self.num_row = self.data.shape[0]
            self.num_columns = self.data.shape[1]


        C,_ = self.almLasso_mat_fun()

        sInd = self.findRep(C, thrS, self.norm_type)

        repInd = self.rmRep(sInd, thrP)


        return repInd,C

if __name__ == '__main__':

    # mc = hdf5storage.loadmat('/home/davidenardone/PySMRS/numpy_data.mat')
    # data = np.asarray(mc['data'])

    # data = np.load('/home/davidenardone/PySMRS/numpy_data.npy')
    data = np.random.rand(1000,5000).astype('float32')
    # np.save('/home/davidenardone/PySMRS/numpy_data', data)



    smrs = SMRS(data=data, alpha=5, norm_type=2,
                verbose=False, thr=10**-7, max_iter=5000,
                affine=True,
                PCA=False, GPU=True)


    rep_ind, C = smrs.smrs()

    print (rep_ind)


