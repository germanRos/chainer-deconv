import re
import os
import numpy as np
from scipy import misc
import ipdb

class DataTransformer(object):
	"""This class implements basic data manipulation
	"""

	def __init__(self, opts):
        	self.opts = opts

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

		return data

	def transformLabel(self, label):
		if(self.opts.has_key('newdims')):
			(H, W) = self.opts['newdims']
			label = misc.imresize(label, (H, W), interp='nearest')

			# labels are initially uint8 but in chainer
			# the void class is -1, so...convertion is required
			label = label.astype(np.int32)
			label = label - 1
		return label

