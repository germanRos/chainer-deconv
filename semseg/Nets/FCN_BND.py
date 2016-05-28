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
class FCN_BND(chainer.Chain):
	"""
	FCN_BND
	"""
	def __init__(self, name, CLASSES, preNet=None):
		super(FCN_BND, self).__init__(
			# params of the model
			#ipdb.set_trace(),

			conv1_1 	= L.Convolution2D(3, 64, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			bn1_1		= L.BatchNormalization(64),
			conv1_2		= L.Convolution2D(64, 64, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			bn1_2		= L.BatchNormalization(64),

			conv2_1 	= L.Convolution2D(64, 128, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			bn2_1		= L.BatchNormalization(128),
			conv2_2		= L.Convolution2D(128, 128, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			bn2_2		= L.BatchNormalization(128),

			conv3_1 	= L.Convolution2D(128, 256, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			bn3_1		= L.BatchNormalization(256),
			conv3_2		= L.Convolution2D(256, 256, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			bn3_2		= L.BatchNormalization(256),
			conv3_3		= L.Convolution2D(256, 256, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			bn3_3		= L.BatchNormalization(256),

			conv4_1 	= L.Convolution2D(256, 512, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			bn4_1		= L.BatchNormalization(512),
			conv4_2		= L.Convolution2D(512, 512, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			bn4_2		= L.BatchNormalization(512),
			conv4_3		= L.Convolution2D(512, 512, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			bn4_3		= L.BatchNormalization(512),

			conv5_1 	= L.Convolution2D(512, 512, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			bn5_1		= L.BatchNormalization(512),
			conv5_2		= L.Convolution2D(512, 512, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			bn5_2		= L.BatchNormalization(512),
			conv5_3		= L.Convolution2D(512, 512, 3, stride=(1, 1), pad=(1, 1), initialW=None, initial_bias=None),
			bn5_3		= L.BatchNormalization(512),
	
			fc6 		= L.Convolution2D(512, 4096, 7, stride=(1, 1), pad=(3, 3), initialW=None, initial_bias=None),
			fc7 		= L.Convolution2D(4096, 4096, 1, stride=(1, 1), pad=(0, 0), initialW=None, initial_bias=None),

			score_fr 	= L.Convolution2D(4096, CLASSES, 1, stride=(1, 1), pad=(0, 0), initialW=None, initial_bias=None),
			score2 		= L.Deconvolution2D(CLASSES, CLASSES, (4,3), stride=(2, 2), pad=(1, 1), initialW=None, initial_bias=None),
			score_pool4 	= L.Convolution2D(512, CLASSES, 1, stride=(1, 1), pad=(0, 0), initialW=None, initial_bias=None),
			score4 		= L.Deconvolution2D(CLASSES, CLASSES, (4,3), stride=(2, 2), pad=(1, 1), initialW=None, initial_bias=None),
			score_pool3 	= L.Convolution2D(256, CLASSES, 1, stride=(1, 1), pad=(0, 0), initialW=None, initial_bias=None),
			upsample 	= L.Deconvolution2D(CLASSES, CLASSES, 8, stride=(8, 8), pad=(0, 0), initialW=None, initial_bias=None),
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
		cblob = self.bn1_1(cblob)
		cblob = F.dropout(cblob, ratio=0.1, train=not test_mode)
		cblob = F.relu(cblob)
		cblob = self.conv1_2(cblob)
		cblob = self.bn1_2(cblob)
		cblob = F.dropout(cblob, ratio=0.1, train=not test_mode)
		cblob = F.relu(cblob)
		cblob = F.max_pooling_2d(cblob, (2, 2), stride=(2,2), pad=(0, 0))

		cblob = self.conv2_1(cblob)
		cblob = self.bn2_1(cblob)
		cblob = F.dropout(cblob, ratio=0.1, train=not test_mode)
		cblob = F.relu(cblob)

		cblob = self.conv2_2(cblob)
		cblob = self.bn2_2(cblob)
		cblob = F.dropout(cblob, ratio=0.1, train=not test_mode)
		cblob = F.relu(cblob)
		cblob = F.max_pooling_2d(cblob, (2, 2), stride=(2,2), pad=(0, 0))

		cblob = self.conv3_1(cblob)
		cblob = self.bn3_1(cblob)
		cblob = F.dropout(cblob, ratio=0.1, train=not test_mode)
		cblob = F.relu(cblob)

		cblob = self.conv3_2(cblob)
		cblob = self.bn3_2(cblob)
		cblob = F.dropout(cblob, ratio=0.1, train=not test_mode)
		cblob = F.relu(cblob)

		cblob = self.conv3_3(cblob)
		cblob = self.bn3_3(cblob)
		cblob = F.dropout(cblob, ratio=0.1, train=not test_mode)
		cblob = F.relu(cblob)
		cblob_pool3 = F.max_pooling_2d(cblob, (2, 2), stride=(2,2), pad=(0, 0))

		cblob = self.conv4_1(cblob_pool3)
		cblob = self.bn4_1(cblob)
		cblob = F.dropout(cblob, ratio=0.1, train=not test_mode)
		cblob = F.relu(cblob)
		cblob = self.conv4_2(cblob)
		cblob = self.bn4_2(cblob)
		cblob = F.dropout(cblob, ratio=0.1, train=not test_mode)
		cblob = F.relu(cblob)
		cblob = self.conv4_3(cblob)
		cblob = self.bn4_3(cblob)
		cblob = F.dropout(cblob, ratio=0.1, train=not test_mode)
		cblob = F.relu(cblob)
		cblob_pool4 = F.max_pooling_2d(cblob, (2, 2), stride=(2,2), pad=(0, 0))

		cblob = self.conv5_1(cblob_pool4)
		cblob = self.bn5_1(cblob)
		cblob = F.dropout(cblob, ratio=0.1, train=not test_mode)
		cblob = F.relu(cblob)

		cblob = self.conv5_2(cblob)
		cblob = self.bn5_2(cblob)
		cblob = F.dropout(cblob, ratio=0.1, train=not test_mode)
		cblob = F.relu(cblob)
		cblob = self.conv5_3(cblob)
		cblob = self.bn5_3(cblob)
		cblob = F.dropout(cblob, ratio=0.1, train=not test_mode)
		cblob = F.relu(cblob)
		cblob = F.max_pooling_2d(cblob, (2, 2), stride=(2,2), pad=(0, 0))

		cblob = self.fc6(cblob)
		cblob = F.relu(cblob)
		cblob = F.dropout(cblob, ratio=0.5, train=not test_mode)

		cblob = self.fc7(cblob)
		cblob = F.relu(cblob)
		cblob = F.dropout(cblob, ratio=0.5, train=not test_mode)

		cblob = self.score_fr(cblob)
		cblob_score2 = self.score2(cblob)
		cblob_score_pool4 = self.score_pool4(cblob_pool4)
		cblob = F.sum_binary(cblob_score2, cblob_score_pool4)

		cblob_score4 = self.score4(cblob)
		cblob_score_pool3 = self.score_pool3(cblob_pool3)
		cblob = F.sum_binary(cblob_score4, cblob_score_pool3)

		self.blob_class = self.upsample(cblob)
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
