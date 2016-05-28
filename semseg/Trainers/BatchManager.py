import re
import os
import math
import numpy as np
from scipy import misc
import chainer
from chainer import cuda
from DataTransformer import DataTransformer

#import ipdb

class BatchManager(object):
	"""This class implements different policies for generating minibatches
	"""

	def __init__(self, xp, policy='krandom', batches_per_epoch=0, batch_size=10, data_transformer=None):
		self.xp = xp
        	self.policy = policy
		self.flist = []
		self.batches_per_epoch = batches_per_epoch
		self.batch_size = batch_size
		self.batches_generated = 0
		self.images_generated = 0
		self.N = 0
		self.data_transformer = data_transformer

		# dims data
		self.H = 0
		self.W = 0
		self.CH = 0

		# dims gt
		self.HL = 0
		self.WL = 0
		self.CHL = 0

		self.weights_classes_flag = False

	def finish_round(self):
		self.batches_generated = 0
		self.images_generated = 0

	def initialize(self, filename):
		pass

	def still_batches(self):
		if(self.policy == 'krandom'):
			return (self.batches_generated != self.batches_per_epoch)
		elif(self.policy == 'sequential'):
			return (self.images_generated < len(self.flist))

	# this method has to be implemented in a child class
    	@classmethod
	def getStatistics(cls, dataaccess, opts):
		pass

	def getRandomSamples(self, K):
		permuted_indices = np.random.permutation(self.N)
		indices = permuted_indices[0:K]
		tup_cpu = self.getBatch_(indices)

		# this is needed in case the getBatch_ returns more than 2 elements (e.g., labels!)
		tup_cpu = (tup_cpu[0], tup_cpu[1])

		tup_end = ()
		for i in range(len(tup_cpu)):
			d = chainer.Variable(self.xp.asarray(tup_cpu[i]))
			tup_end += (d,)
		return tup_end

	
	def generateBatchIndices(self):
		if(self.policy == 'krandom'):
			permuted_indices = np.random.permutation(self.N)
			return permuted_indices[0:self.batch_size]	
		elif(self.policy == 'sequential'):
			return np.array(range(self.images_generated, min(self.images_generated+self.batch_size, self.N)))

	def getDimsData(self):
		return self.H, self.W, self.CH

	def getDimsLabel(self):
		return self.HL, self.WL, self.CHL

	def getBatchesGenerated(self):
		return self.batches_generated

	def getNBatchPrediction(self):
		if(self.policy == 'krandom'):
			return self.batches_per_epoch
		elif(self.policy == 'sequential'):
			return np.uint32(math.ceil(float(self.N) / float(self.batch_size)))	

	def getBatch(self):
		# first indices are generated according with the selected policy
		indices = self.generateBatchIndices()
		
		tup_cpu = self.getBatch_(indices)
		tup_end = ()
		for i in range(len(tup_cpu)):
			d = chainer.Variable(self.xp.asarray(tup_cpu[i]))
			tup_end += (d,)


		self.batches_generated += 1
		self.images_generated += self.batch_size
		return tup_end

	# this method has to be implemented in a child class
	def getBatch_(self, indices):
		pass
		
