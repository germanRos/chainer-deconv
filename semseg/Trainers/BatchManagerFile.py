from BatchManager import BatchManager
import re
import os
import numpy as np
from scipy import misc
import ipdb

class BatchManagerFile(BatchManager):
	"""This class deals with batches from disk
	"""

	def initialize(self, filename):
		# read filename
		with open(filename) as f:
			for line in f:
				cline = re.split('\s',line)
				# tuple (rgb_filename, gt_filename)
				self.flist.append((cline[0], cline[1]))
		self.N = len(self.flist)

		# get sizes
		(rgbname, gtname) = self.flist[0]
		rgb = misc.imread(rgbname)

		if(gtname.endswith('.png')):
			gt = misc.imread(gtname)
		else:
			gt = np.loadtxt(gtname)
		(self.H, self.W, self.CH) = rgb.shape
		gtshape = gt.shape
		self.HL = gtshape[0]
		self.WL = gtshape[1]
		if(len(gtshape) == 2):
			self.CHL = 1
		else:
			self.CHL = gtshape[2]

		# just retrieve the new sizes
		if( (self.data_transformer is not None) and self.data_transformer.isResized() ):
			(self.H, self.W) = self.data_transformer.getNewDims()
			self.HL = self.H
			self.WL = self.W
		
    	@classmethod
	def getStatistics(cls, filename, opts):
		# read filename
		NCLASSES = opts['num_classes']
		rangeMin = opts['stats_range_min']
		rangeMax = opts['stats_range_max']

		stats_ = np.zeros((NCLASSES, 1), dtype=np.float32)
		mean_ = np.ndarray((0), dtype=np.float32)
		min_ = np.ndarray((0), dtype=np.float32)
		max_ = np.ndarray((0), dtype=np.float32)
		with open(filename) as f:
			num_images = 0
			for line in f:
				cline = re.split('\s',line)
				print(cline)
				rgb = misc.imread(cline[0])

				if(cline[1].endswith('.png')):
					gt = misc.imread(cline[1])
				else:
					gt = np.loadtxt(cline[1])
				gt = gt.astype(np.int32)
				gt = gt - 1

				# accumulate values for stats
				for c in range(NCLASSES):
					stats_[c] += (gt == c).sum()

				if(mean_.size == 0):
					mean_ = rgb.mean((0,1)).astype(np.float32)
					min_ = rgb.min((0,1)).astype(np.float32)
					max_ = rgb.max((0,1)).astype(np.float32)
				else:
					mean_ += rgb.mean((0,1)).astype(np.float32)
					min_ = np.minimum(min_, rgb.min((0,1)).astype(np.float32))
					max_ = np.maximum(max_, rgb.max((0,1)).astype(np.float32))
				num_images += 1
			mean_ /= float(num_images)	

			stats_ = stats_.sum() / (stats_+0.1)
			leftMax = stats_.max()
			leftMin = stats_.min()
			rightMax = rangeMax
			rightMin = rangeMin

			leftSpan = leftMax - leftMin
			rightSpan = rightMax - rightMin
			valueScaled = (stats_ - leftMin) / float(leftSpan)
			stats_ = rightMin + (valueScaled * rightSpan)
			
		return {'mean':mean_, 'min':min_, 'max':max_, 'weights':stats_}

	def set_weights_classes(self, weights):
		self.weights_classes = weights
		self.weights_classes_flag = True	

	def getBatch_(self, indices):
		# format NxCHxWxH
		batchRGB = np.zeros((len(indices), self.CH, self.W, self.H), dtype='float32')
		batchLabel = np.zeros((len(indices), self.W, self.H), dtype='int32')

		k = 0
		for i in indices:
			(rgbname, gtname) = self.flist[i]

			# format: HxWxCH
			rgb =  misc.imread(rgbname)

			if(gtname.endswith('.png')):
				gt = misc.imread(gtname)
			else:
				gt = np.loadtxt(gtname)
			gt = gt.astype('uint8')
		
			if(self.data_transformer is not None):
				rgb = self.data_transformer.transformData(rgb)
				gt = self.data_transformer.transformLabel(gt)
			#^ data_transformer outputs in format HxWxCH

			# convertion from HxWxCH to CHxWxH
			batchRGB[k,:,:,:] = rgb.astype(np.float32).transpose((2,1,0))
			batchLabel[k,:,:] = gt.astype(np.int32).transpose((1,0))

			k += 1

			#ipdb.set_trace()

		if(self.weights_classes_flag):
			return (batchRGB, batchLabel, self.weights_classes)
		else:
			return (batchRGB, batchLabel)

