from SMRS import SMRS
import numpy as np
np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':

    n_feats = 100
    n_sample = 300

    Y = np.random.rand(n_feats,n_sample)

    print ('Extracting the representatives frames from the video...It may takes a while...')
    smrs = SMRS(data=Y, alpha=5,norm_type=2,
                thr=10**-8, max_iter=5000, affine=True,
                verbose=False,PCA=True)

    rep_ind, C = smrs.smrs()

    print ('The representatives are:')
    print (rep_ind)