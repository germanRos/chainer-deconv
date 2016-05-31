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
import cv2
import ipdb

# the model we want to train
from Nets.FCN_BND import FCN_BND as Netmodel


GPU_ID				= 0
CLASSES				= 11
NEWSIZE				= (360, 480)
MODEL_NAME			= '/Extra/Experiments/tmp_FCNBND_e1_/FCNBND_e1__392.model'
TESTFILE			= '/Extra/Data/Images/list.txt'

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
	cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
	#cv2.namedWindow('pred', cv2.WINDOW_NORMAL)


	# load model
	model = Netmodel('eval-model', CLASSES)
	serializers.load_npz(MODEL_NAME, model)
	cuda.get_device(GPU_ID).use()
	model.to_gpu()

	LUT = fromHEX2RGB(stats_opts['colormap'] )
	fig3, axarr3 = plt.subplots(1, 1)

	batchRGB = np.zeros((1, 3, NEWSIZE[1], NEWSIZE[0]), dtype='float32')

	# go throught the data
	flist = []
	with open(TESTFILE) as f:
		for line in f:
			cline = re.split('\n',line)

			#print(cline[0])
			frame = misc.imread(cline[0])
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		
			# process frame
			im = misc.imresize(frame, NEWSIZE, interp='bilinear')
			# convertion from HxWxCH to CHxWxH
			batchRGB[0,:,:,:] = im.astype(np.float32).transpose((2,1,0))
			batchRGBn = batchRGB  - 127.0

			# data ready
			batch = chainer.Variable(cuda.cupy.asarray(batchRGBn))

			# make predictions
			model((batch, []), test_mode=2)
			pred = model.probs.data.argmax(1)
			# move data back to CPU
			pred_ = cuda.to_cpu(pred)

			pred_ = LUT[pred_+1,:].squeeze()
			pred_ = pred_.transpose((1,0,2))
			pred2 = cv2.cvtColor(pred_, cv2.COLOR_BGR2RGB)

			#ipdb.set_trace()
			disp = (0.4*im + 0.6*pred2).astype(np.uint8)

    			# Display the resulting frame
   			cv2.imshow('frame',disp)
   			#cv2.imshow('pred',pred2)
    			if cv2.waitKey(-1) & 0xFF == ord('q'):
        			break
	cv2.destroyAllWindows()

inference()
