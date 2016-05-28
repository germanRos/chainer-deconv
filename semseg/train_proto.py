%load_ext autoreload
%autoreload 2

import os
import pickle
import chainer
from chainer import cuda
from Trainers.basicTrainer import BTrainer
from Trainers.BatchManagerFile import BatchManagerFile
from Trainers.DataTransformer import DataTransformer
from Trainers.StatisticsManagerThread import StatisticsManagerThread as StatManager
import Tools.loadMatConvNetModel as lm
import ipdb

# the model we want to train
from Nets.TinynetUnpooling2 import TinynetUnpooling2 as Netmodel
#from Nets.Tinynet4 import Tinynet4 as Netmodel

GPU_ID				= 0
BSIZE				= 10
BATCHES_EPOCH			= 50
NEPOCHS 			= 10000
TRAINFILE			= '/home/gros/Data/pascal_context_11/chainer_train.txt' #'/home/gros/Data/chainer_train_multi.txt'
VALIDFILE			= '/home/gros/Data/chainer_val_multi.txt'
CLASSES				= 11
NEWSIZE				= (360, 480)
CONTINUE 			= True
SAVE_MODEL_EVERY		= 10
SAVE_FOLDER			= 'tmp_'
MODEL_NAME			= 'TinyNet_Unpool4'
DATASET_STATS			= SAVE_FOLDER+MODEL_NAME+'/dataset_stats.pick'
PRE_MODEL_PATH			= '/home/gros/tinynet-red.mat'

# first get dataset global statistics required by the data transformer
if(not os.path.exists(SAVE_FOLDER+MODEL_NAME)):
	os.makedirs(SAVE_FOLDER+MODEL_NAME)
if(not os.path.exists(DATASET_STATS)):
	dataset_opts = {}
	dataset_opts['num_classes'] = CLASSES
	dataset_opts['stats_range_min'] = 1.0
	dataset_opts['stats_range_max'] = 50.0
	dataset_stats 	= BatchManagerFile.getStatistics(TRAINFILE, dataset_opts)
	pickle.dump(dataset_stats, open(DATASET_STATS, "wb"), pickle.HIGHEST_PROTOCOL)
else:
	dataset_stats = pickle.load(open(DATASET_STATS, "rb"))

# Now let's define the rules for image normalization
dtrans_opts = {}
dtrans_opts['resize'] 		= True
dtrans_opts['newdims'] 		= NEWSIZE
dtrans_opts['zeromean'] 	= True
dtrans_opts['dataset_mean'] 	= dataset_stats['mean']
dtrans_opts['rangescale'] 	= True
dtrans_opts['dataset_min'] 	= dataset_stats['min']
dtrans_opts['dataset_max'] 	= dataset_stats['max']
dtrans = DataTransformer(dtrans_opts)

ipdb.set_trace()
# We can now initialize the batch managers with the correct transformations
bmanager_train 	= BatchManagerFile(cuda.cupy, 'krandom', BATCHES_EPOCH, BSIZE, data_transformer=dtrans)
bmanager_train.initialize(TRAINFILE)
bmanager_train.set_weights_classes(dataset_stats['weights'])

bmanager_val  	= BatchManagerFile(cuda.cupy, 'krandom', 5, BSIZE, data_transformer=dtrans)
bmanager_val.initialize(VALIDFILE)
bmanager_val.set_weights_classes(dataset_stats['weights'])

# Stats manager is used for stats ploting and visualization (probably the most important part!)
stats_opts = {}
stats_opts['labels'] 		= ['sky', 'building', 'road', 'sidewalk', 'fence', 'vegetation', 'pole', 'car', 'sign', 'pedestrian', 'cyclist']
stats_opts['colormap'] 		= ['#808080', '#800000', '#804080', '#0000c0', '#404080', '#808000', '#c0c080', '#400080', '#c08080', '#404000', '#0080c0']
stats_opts['bmanagers'] 	= [bmanager_train, bmanager_val]
stats_opts['samples.vis'] 	=  [2, 2] # random samples from each bmanager 
stats_opts['policy'] 		= 'fixed' # 'fixed' or 'variable'
smanager = StatManager(stats_opts)

# Model instantation and the training starts!
pre_model = lm.loadMatConvNetModelClassic(PRE_MODEL_PATH)

model = Netmodel(MODEL_NAME, CLASSES, pre_model)
trainer = BTrainer(model, GPU_ID, CONTINUE)
trainer.train(bmanager_train, bmanager_val, smanager, NEPOCHS, outputFolder=SAVE_FOLDER+MODEL_NAME, saveEach=SAVE_MODEL_EVERY)

