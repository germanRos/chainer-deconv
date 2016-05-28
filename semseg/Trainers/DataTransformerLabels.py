import re
import os
import numpy as np
from scipy import misc
import ipdb

class DataTransformerLabels(object):
	"""This class implements basic data manipulation
	"""

	def __init__(self, opts):
        	self.opts = opts
		self.dataflip_state = False		

	def isResized(self):
		return self.opts.has_key('resize')

	def getNewDims(self):
		if(self.opts.has_key('newdims')):
			return self.opts['newdims']
		else:
			return (0, 0)

	def transformData(self, data):
		if(self.opts.has_key('newdims')):
			(H, W) = self.opts['newdims']
			data = misc.imresize(data, (H, W), interp='bilinear')

		if(self.opts.has_key('zeromean') and self.opts['zeromean']):
			mean = self.opts['dataset_mean'] # provided by bmanager
			data = data - mean


		if(self.opts.has_key('rangescale') and self.opts['rangescale']):
			min_ = self.opts['dataset_min']  # provided by bmanager
			min_ = np.abs(min_.min())
			max_ = self.opts['dataset_max']  # provided by bmanager
			max_ = np.abs(max_.max())
			data = 127 * data / max(min_, max_)
		else:
			data = data - 127.0

		if(self.opts.has_key('randomflip') and self.opts['randomflip']):
			if(np.random.rand() <= self.opts['randomflip_prob']):
				data = np.flipud(data)
				self.dataflip_state = True

		return data

	def transformLabel(self, label):
		if(self.opts.has_key('randomflip') and self.opts['randomflip']):
			if(self.dataflip_state):
				label = np.flipud(label)
				self.dataflip_state = False

		if(self.opts.has_key('newdims')):
			(H, W) = self.opts['newdims']
			label = misc.imresize(label, (H, W), interp='nearest')

		if(self.opts.has_key('mapLabels') and self.opts['mapLabels']):
			map_ = self.opts['lmap']
			label2 = label.copy()
			for i in map_:
				indices = (label == i[0])
				label2[indices] = i[1]
			label = label2
			#ipdb.set_trace()
			#pass

		

		# labels are initially uint8 but in chainer
		# the void class is -1, so...convertion is required
		label = label.astype(np.int32)
		label = label - 1
		return label

