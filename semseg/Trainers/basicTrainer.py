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
import re
import time
import ipdb



class BTrainer(object):
	def __init__(self, model, GPU_ID=0, cont=False):
		self.model = model
		self.cont = cont

		# moving model to default GPU
		cuda.get_device(GPU_ID).use()
		model.to_gpu()

	def train(self, bmanagerT, bmanagerV, smanager, NEpochs, outputFolder='.', saveEach=1):
                optimizer = optimizers.Adam()
                optimizer.setup(self.model)
		#optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

		# continuing from a previous training?
		if(self.cont):
			(self.model, optimizer, last_epoch) = self.loadInfo(outputFolder, self.model, optimizer, smanager)
		else:
			# Setup optimizer
                	optimizer.setup(self.model)
			last_epoch = 1
	
		# main loop
		for epoch in six.moves.range(last_epoch, NEpochs + 1):
			# Training Loop
			smanager.set_mode('train')
			while(bmanagerT.still_batches()):
				tuple_input = bmanagerT.getBatch()
				optimizer.update(self.model, tuple_input)
				# print partial stats
				smanager.update(self.model, tuple_input, epoch, bmanagerT.getBatchesGenerated(), bmanagerT.getNBatchPrediction())
				smanager.print_batch_stats()
			bmanagerT.finish_round()

			# Validation Loop
			smanager.set_mode('valid')
			while(bmanagerV.still_batches()):
				tuple_input = bmanagerV.getBatch()

				start = time.time()
				self.model(tuple_input, test_mode=True)
				stop = time.time()
				print('------ ' + str((stop-start)*1000.0))

				# print partial stats
                                smanager.update(self.model, tuple_input, epoch, bmanagerV.getBatchesGenerated(), bmanagerV.getNBatchPrediction())
                                smanager.print_batch_stats()
			bmanagerV.finish_round()

			# get global stats
			smanager.show_epoch_stats()
			smanager.reset()

			# Save info
			self.saveInfo(self.model, optimizer, smanager, epoch, outputFolder, saveEach)

	def loadInfo(self, folder, model, state, smanager):
		if(not os.path.exists(folder)):
			return (model, state, 1)
		list_files = []
		model_name = model.getName()
		for file in os.listdir(folder):
			if(file.startswith(model_name) and file.endswith(".state")):
				list_files.append(file)
		if(len(list_files) > 0):
			sorted_list = self.natural_sort(list_files)
			fname_state = sorted_list[-1]

			bname = re.split('\.',fname_state)[0]
			fname_model = bname + '.model'
			fname_stats = bname + '.stats'
			epoch = int(re.split('_|\.', bname)[-1]) + 1
			serializers.load_npz(folder + '/' + fname_state, state)
			serializers.load_npz(folder + '/' + fname_model, model)
			smanager.load(folder + '/' + fname_stats)
			
		else:
			epoch = 1
			# no prev. models...
		return (model, state, epoch)

	
	def natural_sort(self, l): 
		convert = lambda text: int(text) if text.isdigit() else text.lower() 
		alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
		return sorted(l, key = alphanum_key)

	def saveInfo(self, model, optimizer, smanager, epoch, outputFolder, saveEach):
		#ipdb.set_trace()
		if(epoch % saveEach == 0):
			if(not os.path.exists(outputFolder)):
 				os.makedirs(outputFolder)
			bname = outputFolder + '/' + model.getName() + '_' + str(epoch)
			serializers.save_npz(bname + '.model', model)
			serializers.save_npz(bname + '.state', optimizer)
			smanager.save(bname + '.stats')

