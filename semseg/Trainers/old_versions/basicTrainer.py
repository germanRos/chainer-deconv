import chainer
import chainer.functions as F
import chainer.links as L
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from BatchManager import BatchManager
import numpy as np
import six
import os
import ipdb



class BTrainer(object):
	def __init__(self, model, GPU_ID=0):
		self.model = model

		# moving model to default GPU
		cuda.get_device(GPU_ID).use()
		model.to_gpu()

	def train(self, bmanagerT, bmanagerV, smanager, NEpochs, outputFolder='.', saveEach=1):
		# Setup optimizer
                optimizer = optimizers.Adam()
                optimizer.setup(self.model)

		# main loop
		for epoch in six.moves.range(1, NEpochs + 1):
			# Training Loop
			smanager.set_mode('train')
			while(bmanagerT.still_batches()):
				(rgb, gt) = bmanagerT.getBatch()
				optimizer.update(self.model, rgb, gt)
				# print partial stats
				smanager.update(self.model, (rgb, gt), epoch, bmanagerT.getBatchesGenerated(), bmanagerT.getNBatchPrediction())
				smanager.print_batch_stats()
			bmanagerT.finish_round()

			# Validation Loop
			smanager.set_mode('valid')
			while(bmanagerV.still_batches()):
				(rgb, gt) = bmanagerV.getBatch()
				self.model(rgb, gt)

				# print partial stats
                                smanager.update(self.model, (rgb, gt), epoch, bmanagerV.getBatchesGenerated(), bmanagerV.getNBatchPrediction())
                                smanager.print_batch_stats()
			bmanagerV.finish_round()

			# get global stats
			smanager.show_epoch_stats()
			smanager.reset()

			# Save info
			self.saveInfo(self.model, optimizer, epoch, outputFolder, saveEach)

	def saveInfo(self, model, optimizer, epoch, outputFolder, saveEach):
		if(epoch % saveEach == 0):
			if(not os.path.exists(outputFolder)):
 				os.makedirs(outputFolder)
			bname = outputFolder + '/' + model.getName() + '_' + str(epoch)
			serializers.save_npz(bname + '.model', model)
			serializers.save_npz(bname + '.state', optimizer)

