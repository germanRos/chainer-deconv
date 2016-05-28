import chainer
import chainer.functions as F
import chainer.links as L
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
import numpy as np
import six
import ipdb


# ------------- Net definition --------------
class FCN(chainer.Chain):
	"""
	TinyNet-v4
	"""
	def __init__(self, name, CLASSES, preNet=None):
		super(FCN, self).__init__(
			# params of the model
			#ipdb.set_trace(),

			conv1_1 	= L.Convolution2D(3, 64, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			conv1_2		= L.Convolution2D(64, 64, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),

			conv2_1 	= L.Convolution2D(64, 128, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			conv2_2		= L.Convolution2D(128, 128, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			
			conv3_1 	= L.Convolution2D(128, 256, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			conv3_2		= L.Convolution2D(256, 256, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			conv3_3		= L.Convolution2D(256, 256, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),			
			
			conv4_1 	= L.Convolution2D(256, 512, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			conv4_2		= L.Convolution2D(512, 512, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			conv4_3		= L.Convolution2D(512, 512, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),			
			
			conv5_1 	= L.Convolution2D(512, 512, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			conv5_2		= L.Convolution2D(512, 512, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			conv5_3		= L.Convolution2D(512, 512, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),	

			#conv_aux	= L.Convolution2D(512, 512, (7, 6), stride=(1, 1), pad=(0, 1), initialW=None, initial_bias=None),
			#conv_auxb	= L.Convolution2D(512, 512, (7, 7), stride=(1, 1), pad=(0, 0), initialW=None, initial_bias=None),

			fc_6		= L.Convolution2D(512, 4096, 1, stride=(1, 1), pad=(0, 0), initialW=None, initial_bias=None),
			fc_7		= L.Convolution2D(4096, 4096, 1, stride=(1, 1), pad=(0, 0), initialW=None, initial_bias=None),
			#fc_6		= L.Linear(512, 4096, wscale=1, bias=0, nobias=False, initialW=None, initial_bias=None),	
			#fc_7		= L.Linear(4096, 4096, wscale=1, bias=0, nobias=False, initialW=None, initial_bias=None),			
			conv_aux2	= L.Convolution2D(4096, CLASSES, 1, stride=(1, 1), pad=(0, 0), initialW=None, initial_bias=None),


			score2  	= L.Deconvolution2D(CLASSES, CLASSES, (4,5), stride=(2, 2), pad=(1, 2)),
			score_pool4	= L.Convolution2D(512, CLASSES, 1, stride=(1, 1), pad=(0, 0), initialW=None, initial_bias=None),

			score4  	= L.Deconvolution2D(CLASSES, CLASSES, (4,3), stride=(2, 2), pad=(1, 1)),
			score_pool3	= L.Convolution2D(256, CLASSES, 1, stride=(1, 1), pad=(0, 0), initialW=None, initial_bias=None),


			upsample 	= L.Deconvolution2D(CLASSES, CLASSES, 8, stride=(8, 8), pad=(0, 0)),
			classi		= L.Convolution2D(CLASSES, CLASSES, 1, stride=(1, 1), pad=(0, 0), initialW=None, initial_bias=None),


		),
		self.name = name
		self.classes = CLASSES
		

	def __call__(self, input_blob, test_mode=False):
		# explicit and very flexible DAG!
		#################################
		data = input_blob[0]
		labels = input_blob[1]

		if(len(input_blob) >= 3):
			weights_classes = input_blob[2]
		else:
			weights_classes = chainer.Variable(cuda.cupy.ones((self.classes, 1), dtype='float32'))

		# ---- CONTRACTION BLOCKS ---- #
		# B1
		#ipdb.set_trace()
		cblob = self.conv1_1(data)
		cblob = F.relu(cblob)
		cblob = self.conv1_2(cblob)
		cblob = F.relu(cblob)
		cblob = F.max_pooling_2d(cblob, (2, 2), stride=(2,2), pad=(0, 0))

		# B2
		cblob = self.conv2_1(cblob)
		cblob = F.relu(cblob)
		cblob = self.conv2_2(cblob)
		cblob = F.relu(cblob)
		cblob = F.max_pooling_2d(cblob, (2, 2), stride=(2,2), pad=(0, 0))

		# B3
		cblob = self.conv3_1(cblob)
		cblob = F.relu(cblob)
		cblob = self.conv3_2(cblob)
		cblob = F.relu(cblob)
		cblob = self.conv3_3(cblob)
		cblob = F.relu(cblob)
		cblob_mp3 = F.max_pooling_2d(cblob, (2, 2), stride=(2,2), pad=(0, 0))

		# B4
		cblob = self.conv4_1(cblob_mp3)
		cblob = F.relu(cblob)
		cblob = self.conv4_2(cblob)
		cblob = F.relu(cblob)
		cblob = self.conv4_3(cblob)
		cblob = F.relu(cblob)
		cblob_mp4 = F.max_pooling_2d(cblob, (2, 2), stride=(2,2), pad=(0, 0))

		# B5
		cblob = self.conv5_1(cblob_mp4)
		cblob = F.relu(cblob)
		cblob = self.conv5_2(cblob)
		cblob = F.relu(cblob)
		cblob = self.conv5_3(cblob)
		cblob = F.relu(cblob)
		cblob = F.max_pooling_2d(cblob, (2, 2), stride=(2,2), pad=(0, 0))
	
		# FCs
		#cblob = self.conv_aux(cblob)	
		#cblob = self.conv_auxb(cblob)	
		cblob = self.fc_6(cblob)
		cblob = F.relu(cblob)
		cblob = F.dropout(cblob, ratio=0.5, train=not test_mode)
		cblob = self.fc_7(cblob)
		cblob = F.relu(cblob)
		cblob = F.dropout(cblob, ratio=0.5, train=not test_mode)
		cblob = self.conv_aux2(cblob)

		# ---- EXPANSION BLOCKS ---- #
		cblob 		= self.score2(cblob)
		cblob_aux1 	= self.score_pool4(cblob_mp4)
		cblob 		= F.sum_binary(cblob, cblob_aux1)
		scoreup 	= self.score4(cblob)

		cblob_aux2	= self.score_pool3(cblob_mp3)
		cblob		= F.sum_binary(scoreup, cblob_aux2)
		cblob		= self.upsample(cblob)

		# ---- SOFTMAX CLASSIFIER ---- #
		self.blob_class = self.classi(cblob)
		self.probs = F.softmax(self.blob_class)

		# ---- WEIGHTED CROSS-ENTROPY LOSS ---- #
		#ipdb.set_trace()
		self.output_point = self.probs

		if(test_mode != 2):
	        	self.loss = F.weighted_cross_entropy(self.probs, labels, weights_classes, normalize=True)
			return self.loss
		else:
			return 0

	def getName(self):
		return self.name
# ------------- Net definition --------------
