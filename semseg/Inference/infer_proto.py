#%load_ext autoreload
#%autoreload 2

import os
import pickle
import math
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from scipy import misc
import chainer
from chainer import serializers
from chainer import cuda
import ipdb

# the model we want to train
from Nets.FCN2 import FCN2 as Netmodel


GPU_ID				= 0
BSIZE				= 1
TESTFILE			= '/DATA/YouTube/comp_all.txt'
CLASSES				= 3
NEWSIZE				= (360, 480)
MODEL_NAME			= '/home/gros/Projects/chainer-1.6.1/semseg/tmp_FCN_nonorm_5classes_e5/FCN_nonorm_5classes_e5_180.model'
OUTPUT 				= '/DATA/Results/Sidewalk'

# Stats manager is used for stats ploting and visualization (probably the most important part!)
stats_opts = {}
stats_opts['labels'] 		= ['sky', 'building', 'road', 'sidewalk', 'fence', 'vegetation', 'pole', 'car', 'sign', 'pedestrian', 'cyclist']
stats_opts['colormap'] 		= ['#808080', '#800000', '#804080', '#0000c0', '#404080', '#808000', '#c0c080', '#400080', '#c08080', '#404000', '#0080c0']

def HTMLColorToRGB(colorstring):
	""" convert #RRGGBB to an (R, G, B) tuple """
	colorstring = colorstring.strip()
	if colorstring[0] == '#': colorstring = colorstring[1:]
	if len(colorstring) != 6:
		raise ValueError, "input #%s is not in #RRGGBB format" % colorstring
	r, g, b = colorstring[:2], colorstring[2:4], colorstring[4:]
	r, g, b = [int(n, 16) for n in (r, g, b)]
	return (r, g, b)


def fromHEX2RGB(colorsM):
	LUT = np.ndarray((len(colorsM)+1, 3), dtype='uint8')
	LUT[0, :] = [0,0,0]
	for i in range(len(colorsM)):
		(r, g, b) = HTMLColorToRGB(colorsM[i])
		LUT[i+1, :] = [r,g,b]
	return LUT

def inference():
	# load model
	model = Netmodel('eval-model', CLASSES)
	serializers.load_npz(MODEL_NAME, model)
	cuda.get_device(GPU_ID).use()
	model.to_gpu()

	LUT = fromHEX2RGB(stats_opts['colormap'] )
	fig3, axarr3 = plt.subplots(1, 1)

	# go throught the data
	flist = []
	with open(TESTFILE) as f:
		for line in f:
			cline = re.split('\n',line)
			flist.append(cline[0])
	N = len(flist)
	H = NEWSIZE[0]
	W = NEWSIZE[1]
	CH = 3

	# load the batches
	ID = 0
	for b in range(np.ceil(N/BSIZE).astype(np.int32)):
		#ipdb.set_trace()

		i0 = b*BSIZE
		iN = min(N, i0+BSIZE)
		batchRGB = np.zeros((iN-i0+1, CH, W, H), dtype='float32')
	
		# fillin batch
		k = 0
		for i in range(i0, iN):
			im = misc.imread(flist[i])
			im = misc.imresize(im, NEWSIZE, interp='bilinear')

			# convertion from HxWxCH to CHxWxH
			batchRGB[k,:,:,:] = im.astype(np.float32).transpose((2,1,0))
			k = k + 1

#		# computing local stats
#		mean_ 	= batchRGB.mean((0,2,3)).astype(np.float32)
#		min_ 	= batchRGB.min((0,2,3)).astype(np.float32)
#		max_	= batchRGB.max((0,2,3)).astype(np.float32)

#		# normalizing batch
#		batchRGBn = batchRGB - mean_[np.newaxis, :, np.newaxis, np.newaxis]
#		min_ = np.abs(min_.min())
#		max_ = np.abs(max_.max())
#		batchRGBn = 127 * batchRGBn / max(min_, max_)

		batchRGBn = batchRGB  - 127.0

		# data ready
		batch = chainer.Variable(cuda.cupy.asarray(batchRGBn))

		# make predictions
		model((batch, []), test_mode=2)

		pred = model.probs.data.argmax(1)
		# move data back to CPU
		pred_ = cuda.to_cpu(pred)

		pred_ = LUT[pred_+1,:].squeeze()
		pred_ = pred_.transpose((0,2,1,3))

		for j in range(k):
			p = pred_[j, :, :, :].squeeze()
			rgb = batchRGB[j, :, :, :].squeeze().transpose((2,1,0))

			p = (0.7*p + 0.3*rgb).astype('uint8')
			misc.imsave(OUTPUT + '/' + '{:06d}'.format(ID) + '.png', p)
			axarr3.imshow(p)
			plt.draw()
			plt.pause(0.01)

			ID = ID + 1

		# done!
		#print('--- done [' + str(b) + '/' + str(np.ceil(N/BSIZE).astype(np.int32)) + ']')

inference()
