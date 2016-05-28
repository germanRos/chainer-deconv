import h5py
import numpy as np
import ipdb

def loadMatConvNetModel(mfile):
	M = h5py.File(mfile, 'r')
	shape = M['net/params/value'].shape[0]

	outDic = {}
	for i in range(shape):
		data = M[M['net/params/value'][i][0]].value
		name = ''.join(chr(i) for i in M[M['net/params/name'][i][0]].value[:])
		outDic[name] = data
	return outDic


def loadMatConvNetModelClassic(mfile):
	M = h5py.File(mfile, 'r')
	num_layers = M['layers'].shape[0]

	outList = []
	for l in range(num_layers):
		ref_layer = M['layers'][l,0]
		layer = M[ref_layer]

		if('weights' not in layer.keys()):
			continue

		num_weights = layer['weights'].shape
		
		# weights
		tuple_weights = ()
		for i in range(num_weights[0]):
			for j in range(num_weights[1]):
				W = M[layer['weights'][i,j]].value
				tuple_weights += (W,)

		# name of the layer
		name = ''.join(chr(k) for k in layer['name'])

		# add both to dict
		outList.append((name, tuple_weights))

	return outList
