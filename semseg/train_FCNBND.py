#%load_ext autoreload
#%autoreload 2
import sys
import os
import pickle
import ipdb

sys.path.insert(0, '/home/adas/Projects/chainer')
sys.path.insert(0, '/home/adas/Projects/chainer/build/lib.linux-x86_64-2.7')
sys.path.insert(0, '')

import chainer
from chainer import cuda
from Trainers.basicTrainer import BTrainer
from Trainers.BatchManagerFile import BatchManagerFile
from Trainers.DataTransformerLabels import DataTransformerLabels
from Trainers.StatisticsManagerThread import StatisticsManagerThread as StatManager
import Tools.loadMatConvNetModel as lm

# the model we want to train
from Nets.FCN_BND import FCN_BND as Netmodel


GPU_ID				= 0
BSIZE				= 4
BATCHES_EPOCH			= 2
NEPOCHS 			= 10000
TRAINFILE			= '/Extra/Data/DataSEMSEG/chainer_total.txt'
VALIDFILE			= '/Extra/Data/DataSEMSEG/chainer_val_multi.txt'
CLASSES				= 11
NEWSIZE				= (360, 480)
CONTINUE 			= True
SAVE_MODEL_EVERY		= 1
SAVE_FOLDER			= '/Extra/Experiments/tmp_'
MODEL_NAME			= 'FCNBND_e1_'
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

dataset_stats['weights'][0][0] = 1.0
dataset_stats['weights'][1][0] = 2.0
dataset_stats['weights'][2][0] = 8.0
dataset_stats['weights'][3][0] = 8.0

print(dataset_stats['weights'])
#ipdb.set_trace()

# Now let's define the rules for image normalization
dtrans_opts = {}
dtrans_opts['resize'] 		= True
dtrans_opts['newdims'] 		= NEWSIZE
dtrans_opts['zeromean'] 	= False
dtrans_opts['dataset_mean'] 	= dataset_stats['mean']
dtrans_opts['rangescale'] 	= False
dtrans_opts['dataset_min'] 	= dataset_stats['min']
dtrans_opts['dataset_max'] 	= dataset_stats['max']
dtrans_opts['mapLabels'] 	= False
dtrans_opts['lmap'] 		= [ [1, 1], [2, 1], [3, 2], [4, 3], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1], [10, 1], [11, 1] ]
dtrans_opts['randomflip'] 	= False
dtrans_opts['randomflip_prob'] 	= 0.4
		

dtrans = DataTransformerLabels(dtrans_opts)

#ipdb.set_trace()
# We can now initialize the batch managers with the correct transformations
bmanager_train 	= BatchManagerFile(cuda.cupy, 'krandom', BATCHES_EPOCH, BSIZE, data_transformer=dtrans)
bmanager_train.initialize(TRAINFILE)
bmanager_train.set_weights_classes(dataset_stats['weights'])

bmanager_val  	= BatchManagerFile(cuda.cupy, 'krandom', 5, BSIZE, data_transformer=dtrans)
bmanager_val.initialize(VALIDFILE)
bmanager_val.set_weights_classes(dataset_stats['weights'])

# Stats manager is used for stats ploting and visualization (probably the most important part!)
stats_opts = {}


stats_opts['labels'] 		= ['others', 'road', 'sidewalk', 'sidewalk', 'fence', 'vegetation', 'pole', 'car', 'sign', 'pedestrian', 'cyclist']
stats_opts['colormap'] 		= ['#808080', '#800000', '#804080', '#0000c0', '#404080', '#808000', '#c0c080', '#400080', '#c08080', '#404000', '#0080c0']

stats_opts['bmanagers'] 	= [bmanager_train, bmanager_val]
stats_opts['samples.vis'] 	=  [2, 2] # random samples from each bmanager 
stats_opts['policy'] 		= 'fixed' # 'fixed' or 'variable'
smanager = StatManager(stats_opts)

# Model instantation and the training starts!
#pre_model = lm.loadMatConvNetModelClassic(PRE_MODEL_PATH)

model = Netmodel(MODEL_NAME, CLASSES)
trainer = BTrainer(model, GPU_ID, CONTINUE)
trainer.train(bmanager_train, bmanager_val, smanager, NEPOCHS, outputFolder=SAVE_FOLDER+MODEL_NAME, saveEach=SAVE_MODEL_EVERY)

