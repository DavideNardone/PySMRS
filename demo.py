from SMRS import SMRS
import numpy as np
np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':

    n_sample = 500
    n_feats = 500

    Y = np.random.rand(n_sample,n_feats)

    print ('Extracting the representatives frames from the video...It may takes a while...')
    smrs = SMRS(data=Y, alpha=5,norm_type=2,
                thr=[10**-8], max_iter=5000, affine=True,
                verbose=False,PCA=True)

    sInd, repInd, C = smrs.smrs()

    print ('The representatives are:')
    print (repInd)
