%load_ext autoreload
%autoreload 2

import os

os.chdir('..')
import matplotlib
import matplotlib.pyplot as plt
from scipy import misc
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import ipdb
from chainer import cuda
from Trainers.BatchManagerFile import BatchManagerFile
from Trainers.DataTransformer import DataTransformer



GPU_ID				= 0
BSIZE				= 10
TRAINFILE			= '/home/gros/Data/chainer_train_multi.txt'
NEWSIZE				= (360, 480)


# Now let's define the rules for image normalization
dtrans_opts = {}
dtrans_opts['resize'] 		= True
dtrans_opts['newdims'] 		= NEWSIZE
dtrans = DataTransformer(dtrans_opts)

# We can now initialize the batch managers with the correct transformations
bmanager_train 	= BatchManagerFile(cuda.cupy, 'krandom', 1, BSIZE, data_transformer=dtrans)
bmanager_train.initialize(TRAINFILE)


(rgb, gt) = bmanager_train.getRandomSamples(BSIZE)
(rgb_pool, indices, size) = F.max_pooling_2dIndices(rgb,  (2, 2), stride=(2,2), pad=(0, 0))

rgb_unpool = F.unpooling_2d(rgb_pool, indices, size)


im1 = rgb.data.transpose(3,2,1,0)[:,:,:,0]
im1_cpu = cuda.cupy.asnumpy(im1)
fig1, ax1 = plt.subplots()
ax1.imshow(im1_cpu)

im2 = rgb_pool.data.transpose(3,2,1,0)[:,:,:,0]
im2_cpu = cuda.cupy.asnumpy(im2)
fig2, ax2 = plt.subplots()
ax2.imshow(im2_cpu)

im3 = rgb_unpool.transpose(3,2,1,0)[:,:,:,0]
im3_cpu = cuda.cupy.asnumpy(im3)
fig3, ax3 = plt.subplots()
ax3.imshow(im3_cpu)

plt.show()
print('Ok, here we are')
print('----')
