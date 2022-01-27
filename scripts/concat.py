import numpy as np


oasis_healthy = np.load('./data/OASIS/healthy_oasis/*.npy')
oasis_adv = np.load('./data/OASIS/adv_oasis/*.npy')
mslub = np.load('./data/MSLUB/*.npy')
brats = np.load('./data/BraTS/*.npy')

oasis_2c = np.concatenate((oasis_adv, oasis_healthy), axis=-1)
np.save('./data/OASIS/2c/*.npy', oasis_2c)

mslub_2c = np.concatenate((mslub, mslub), axis=-1)
np.save('./data/MSLUB/*_2c.npy', mslub_2c)

brats_2c = np.concatenate((brats, brats), axis=-1)
np.save('./data/BraTS/*_2c.npy', brats_2c)
