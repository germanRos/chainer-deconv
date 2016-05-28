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
class TinynetUnpooling(chainer.Chain):
	"""
	TinyNet-v4
	"""
	def __init__(self, name, CLASSES):
		super(TinynetUnpooling, self).__init__(
			# params of the model
			bnorm0  = L.BatchNormalization(3),

			conv1 	= L.Convolution2D(3, 64, 7, stride=(1, 1), pad=(3, 3)),
			bnorm1	= L.BatchNormalization(64),
			
			conv2	= L.Convolution2D(64, 64, 7,  stride=(1, 1), pad=(3, 3)),
			bnorm2	= L.BatchNormalization(64),

			conv3	= L.Convolution2D(64, 64, 7,  stride=(1, 1), pad=(3, 3)),
			bnorm3	= L.BatchNormalization(64),

			conv4	= L.Convolution2D(64, 64, 7,  stride=(1, 1), pad=(3, 3)),
			bnorm4	= L.BatchNormalization(64),

			conv5	= L.Convolution2D(64, 64, 7,  stride=(1, 1), pad=(3, 3)),
			bnorm5	= L.BatchNormalization(64),

			conv6	= L.Convolution2D(64, 64, 7,  stride=(1, 1), pad=(3, 3)),
			bnorm6	= L.BatchNormalization(64),

			conv7	= L.Convolution2D(64, 64, 7,  stride=(1, 1), pad=(3, 3)),
			bnorm7	= L.BatchNormalization(64),

			conv8	= L.Convolution2D(64, 64, 7,  stride=(1, 1), pad=(3, 3)),
			bnorm8	= L.BatchNormalization(64),

			classi 	= L.Convolution2D(64, CLASSES, 7, stride=(1, 1), pad=(3, 3)),

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
		blob_b0  = self.bnorm0(data)
		(blob_b1, indices_b1, size_b1)  = F.max_pooling_2dIndices(self.bnorm1(F.relu(self.conv1(blob_b0)), test=test_mode), (2, 2), stride=(2,2), pad=(0, 0))
		(blob_b2, indices_b2, size_b2)  = F.max_pooling_2dIndices(self.bnorm2(F.relu(self.conv2(blob_b1)), test=test_mode), (2, 2), stride=(2,2), pad=(0, 0))
		(blob_b3, indices_b3, size_b3)  = F.max_pooling_2dIndices(self.bnorm3(F.relu(self.conv3(blob_b2)), test=test_mode), (2, 2), stride=(2,2), pad=(0, 0))
		(blob_b4, indices_b4, size_b4)  = F.max_pooling_2dIndices(self.bnorm4(F.relu(self.conv4(blob_b3)), test=test_mode), (2, 2), stride=(2,2), pad=(0, 0))

		# ---- EXPANSION BLOCKS ---- #
		blob_b5  = self.bnorm5(F.relu(self.conv5(F.unpooling_2d(blob_b4, indices_b4, size_b4))), test=test_mode)
		blob_b6  = self.bnorm6(F.relu(self.conv6(F.unpooling_2d(blob_b5, indices_b3, size_b3))), test=test_mode)
		blob_b7  = self.bnorm7(F.relu(self.conv7(F.unpooling_2d(blob_b6, indices_b2, size_b2))), test=test_mode)
		blob_b8  = self.bnorm8(F.relu(self.conv8(F.unpooling_2d(blob_b7, indices_b1, size_b1))), test=test_mode)

		#ipdb.set_trace()

		# ---- SOFTMAX CLASSIFIER ---- #
		self.blob_class = self.classi(blob_b8)
		self.probs = F.softmax(self.blob_class)

		# ---- CROSS-ENTROPY LOSS ---- #
		#ipdb.set_trace()
	        self.loss = F.weighted_cross_entropy(self.probs, labels, weights_classes, normalize=True)
		self.output_point = self.probs

		return self.loss

	def getName(self):
		return self.name
# ------------- Net definition --------------
