import re
import os
import numpy as np
from scipy import misc
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from scipy.interpolate import spline
from chainer import cuda
import pickle
import ipdb

def HTMLColorToRGB(colorstring):
	""" convert #RRGGBB to an (R, G, B) tuple """
	colorstring = colorstring.strip()
	if colorstring[0] == '#': colorstring = colorstring[1:]
	if len(colorstring) != 6:
		raise ValueError, "input #%s is not in #RRGGBB format" % colorstring
	r, g, b = colorstring[:2], colorstring[2:4], colorstring[4:]
	r, g, b = [int(n, 16) for n in (r, g, b)]
	return (r, g, b)

def fromHEX2RGB(colorsM):
	LUT = np.ndarray((len(colorsM)+1, 3), dtype='uint8')
	LUT[0, :] = [0,0,0]
	for i in range(len(colorsM)):
		(r, g, b) = HTMLColorToRGB(colorsM[i])
		LUT[i+1, :] = [r,g,b]
	return LUT

class Stats(object):
	def __init__(self):
		self.confm_list = []
		self.global_acc_batch = 0
		self.avg_perclass_batch = 0
		self.loss_batch = 0
		self.epoch = []
		self.batch = 0
		self.loss = []
		self.global_acc = []
		self.avg_perclass = []
		self.totalBatches = 0
	# --------------------------------:

	def reset(self):
		self.global_acc_batch = 0
		self.avg_perclass_batch = 0
		self.loss_batch = 0
		self.batch = 0
		self.totalBatches = 0
	# --------------------------------:


class StatisticsManager(object):
	"""This class implements basic statistics computation for semantic segmentation
	"""

	def __init__(self, opts):
		#ion() 
        	self.opts = opts
		self.mode = None
		self.dic_stats = {}
		self.debug = False
		self.labels = self.opts['labels']
		self.colorsM = self.opts['colormap']
		self.LUT = fromHEX2RGB(self.colorsM)
		self.L = len(self.labels)
		self.last_net = []

		self.listManagers = self.opts['bmanagers']
		self.num_random_imgs = self.opts['samples.vis']
		self.policy = self.opts['policy']

		self.listSampleImages = []
		if(self.policy == 'fixed'):
			for i in range(len(self.listManagers)):
				self.listSampleImages.append(self.listManagers[i].getRandomSamples(self.num_random_imgs[i]))
		

		# visualization of loss + errors
		self.fig1, self.axarr = plt.subplots(1, 2)
		self.l1, = self.axarr[0].plot([],[], color='#0072bd', linestyle='-', linewidth=1.5, label='loss train')
		self.l2, = self.axarr[0].plot([],[], color='#d95319', linestyle='--', linewidth=1.5, label='loss val')
		self.axarr[0].grid()
		self.axarr[0].legend(loc='lower left')
		self.axarr[0].set_xlabel('epoch')
		self.axarr[0].set_ylabel('loss')
		self.axarr[0].set_xlim((1, 1))
		self.axarr[0].set_ylim((0, 1000))

		self.l3, = self.axarr[1].plot([],[], color='#0072bd', linestyle='-', linewidth=1.5, label='global train')
		self.l4, = self.axarr[1].plot([],[], color='#edb121', linestyle='--', linewidth=1.5, label='global val')
		self.l5, = self.axarr[1].plot([],[], color='#d95319', linestyle='-', linewidth=1.5, label='perclass train')
		self.l6, = self.axarr[1].plot([],[], color='#7e2f8e', linestyle='--', linewidth=1.5, label='perclass val')
		self.axarr[1].grid()
		self.axarr[1].legend(loc='upper left')
		self.axarr[1].set_xlabel('epoch')
		self.axarr[1].set_ylabel('accuracy')
		self.axarr[1].set_xlim((1, 1))
		self.axarr[1].set_ylim((0, 1))

		# visualization of conf matrices
		self.fig2 = plt.figure()
		self.axarr2 = []
		self.axarr2.append(self.fig2.add_axes([0.16, 0.1, 0.40, 0.98])) 
		self.axarr2.append(self.fig2.add_axes([0.57, 0.1, 0.40, 0.98])) 

		# define the legend
		norm = mc.Normalize(0, 1)
		s_m = matplotlib.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
		s_m.set_array([])
		#cb = self.fig2.colorbar(s_m, orientation='horizontal')

		self.fig2.subplots_adjust(bottom=0.8)
		cbar_ax = self.fig2.add_axes([0.02, 0.05, 0.96, 0.06])
		cb = self.fig2.colorbar(s_m, cax=cbar_ax, orientation='horizontal')

		# set conf-matrix training
		self.data_conf_train = self.axarr2[0].imshow(np.zeros((self.L, self.L)), cmap=plt.cm.jet, norm=mc.Normalize(vmin=0.0, vmax=1.0, clip=False), interpolation='nearest')
		self.array_text_training = []
		for x in xrange(self.L):
			for y in xrange(self.L):
				self.array_text_training.append(self.axarr2[0].text(y, x, str("{:.2f}".format(0)), verticalalignment='center', horizontalalignment='center', color='white', fontsize=7, weight='bold'))
		self.axarr2[0].set_title('Conf. Matrix Training', fontsize=12, weight='bold')
		self.axarr2[0].set_xlabel('Estimated', fontsize=9, weight='bold')
		self.axarr2[0].set_ylabel('Expected', fontsize=9, weight='bold')
		self.axarr2[0].set_xticks(range(0, self.L))
		self.axarr2[0].set_yticks(range(0, self.L))

		self.axarr2[0].set_yticklabels(self.labels)
		i = 0
		for label in self.axarr2[0].yaxis.get_ticklabels():
			label.set_color(self.colorsM[i])
			label.set_weight('bold')
			i += 1

		self.axarr2[0].set_xticklabels(self.labels)
		i = 0
		for label in self.axarr2[0].xaxis.get_ticklabels():
			label.set_rotation(90)
			label.set_color(self.colorsM[i])
			label.set_weight('bold')
			i += 1


		# set conf-matrix validation
		self.data_conf_valid = self.axarr2[1].imshow(np.zeros((self.L, self.L)), cmap=plt.cm.jet, norm=mc.Normalize(vmin=0.0, vmax=1.0, clip=False), interpolation='nearest')
		self.array_text_validation = []
		for x in xrange(self.L):
			for y in xrange(self.L):
				self.array_text_validation.append(self.axarr2[1].text(y, x, str("{:.2f}".format(0)), verticalalignment='center', horizontalalignment='center', color='white', fontsize=7, weight='bold'))
		self.axarr2[1].set_title('Conf. Matrix Validation', fontsize=12, weight='bold')
		self.axarr2[1].set_xlabel('Estimated', fontsize=9, weight='bold')
		#self.axarr2[1].set_ylabel('Expected')
		self.axarr2[1].set_xticks(range(0, self.L))
		self.axarr2[1].set_yticks([])

		#self.axarr2[1].set_yticklabels(self.labels)
		self.axarr2[1].set_xticklabels(self.labels)
		i = 0
		for label in self.axarr2[1].xaxis.get_ticklabels():
			label.set_rotation(90)
			label.set_color(self.colorsM[i])
			label.set_weight('bold')
			i += 1
			#ipdb.set_trace()
		

		# showing random images
		self.fig3, self.axarr3 = plt.subplots(len(self.listManagers), 1)
		for i in self.axarr3:
			i.set_xticks([])
			i.set_yticks([])
	# --------------------------------:

	def save(self, fname):
		fd = open(fname,'w') 
		pickle.dump(self.dic_stats, fd)   
		fd.close()

	# --------------------------------:
	def load(self, fname):
		fd = open(fname,'r') 
		self.dic_stats = pickle.load(fd)
		fd.close()
	# --------------------------------:

	def set_mode(self, mode):
		self.mode = mode
		if(mode not in self.dic_stats):
			self.dic_stats[mode] = Stats()
	# --------------------------------:

	def reset(self):
		for key, value in self.dic_stats.iteritems():
			value.reset()
	# --------------------------------:

	def update(self, net, input_blob, epoch, batch, totalBatches):
		self.last_net = net
		stats = self.dic_stats[self.mode]
		stats.batch = batch
		stats.totalBatches = totalBatches
		stats.loss_batch = np.nan_to_num(float(net.loss.data))

		# get the probabilities and turn them into predictions
		probs = net.output_point.data
		pred = probs.argmax(1)
		gt = input_blob[1].data

		NCLASSES = probs.shape[1] 

		#compute conf-matrix to generate all the remaining stats
		local_confm, global_acc, avg_perclass = self.getStats(pred, gt, NCLASSES)

		# update stats
		if(batch > 1):
			stats.confm_list[-1] += local_confm
		else:
			stats.confm_list.append(local_confm)			

		# per-batch stats
		stats.global_acc_batch = global_acc
		stats.avg_perclass_batch = avg_perclass

		# global stats
		global_acc_epoch, avg_perclass_epoch = self.getAccuracies(stats.confm_list[-1])

		# first batch of the epoch...
		if(batch == 1):
			stats.epoch.append(epoch)
			stats.loss.append(stats.loss_batch/float(totalBatches))
			stats.global_acc.append(global_acc_epoch)
			stats.avg_perclass.append(avg_perclass_epoch)
		else:
			stats.loss[-1] 		+= (stats.loss_batch/float(totalBatches))
			stats.global_acc[-1] 	= global_acc_epoch
			stats.avg_perclass[-1] 	= avg_perclass_epoch


		self.dic_stats[self.mode] = stats
	# --------------------------------:

	def getAccuracies(self, confm):
		confm = np.nan_to_num(confm)		
		
		# global acc.
		global_acc = float(np.diag(confm).sum() + 1e-5) / float(confm.sum())

		# normalization
		avg_perclass = 0
		for r_gt in range(0, confm.shape[0]):
			avg_perclass += confm[r_gt, r_gt]/float(confm[r_gt, :].sum() + 1e-5)
		avg_perclass /= float(confm.shape[0])

		return global_acc, avg_perclass
	# --------------------------------:

	def getStats(self, pred, gt, L):
		confm = np.zeros((L,L))

		avg_perclass = 0
		for r_gt in range(0, L):
			gt_idx = (gt == r_gt)
			for c_pred in range(0, L):
				pred_idx = (pred == c_pred)
				# how many hits?
				confm[r_gt, c_pred] = gt_idx.__and__(pred_idx).sum()
			avg_perclass += confm[r_gt, r_gt]/float(confm[r_gt, :].sum() + 1e-5)
		avg_perclass /= float(L)		
		# global acc.
		global_acc = float(confm.diagonal().sum() + 1e-5) / float(confm.sum())

		return confm, global_acc, avg_perclass
	# --------------------------------:

	def print_batch_stats(self):
		stats = self.dic_stats[self.mode]
		print '[mode = ' + self.mode + ' | epoch = ' + '{:08d}'.format(stats.epoch[-1]) + ' | batch = ' + '{:08d}'.format(stats.batch) \
		+ '/' + '{:08d}'.format(stats.totalBatches) + ' | loss = ' + '{:.02f}'.format(stats.loss_batch) \
		+ ' | global acc. = ' + '{:.03f}'.format(stats.global_acc_batch) \
		+ ' | avg. perclass = ' + '{:.03f}'.format(stats.avg_perclass_batch) + ']'
	# --------------------------------:

	def show_epoch_stats(self):
		# This is gonna take all the modes and plot them
		st_train = self.dic_stats['train']
		st_valid = self.dic_stats['valid']

		# first loss plot for training and validation
		plt.figure(self.fig1.number)
		self.l1.set_data(np.array(st_train.epoch), np.array(st_train.loss))
		self.l2.set_data(np.array(st_valid.epoch), np.array(st_valid.loss))

		self.axarr[0].set_xlim((1, np.uint32(st_train.epoch[-1])))
		self.axarr[0].set_ylim((0, max(max(st_train.loss), max(st_valid.loss))))
		plt.pause(.0001)

		# accuracies plot for training and validation
		print('-----> [Training: global acc. = ' + '{:.03f}'.format(st_train.global_acc[-1]) + ' | perclass acc. = ' + '{:.03f}'.format(st_train.avg_perclass[-1]) \
		+ '] [Validation: global acc. = ' + '{:.03f}'.format(st_valid.global_acc[-1]) + ' | perclass acc. = ' + '{:.03f}'.format(st_valid.avg_perclass[-1]) + '] <-----\n') 

		self.l3.set_data(np.array(st_train.epoch), np.array(st_train.global_acc))
		self.l4.set_data(np.array(st_valid.epoch), np.array(st_valid.global_acc))
		self.l5.set_data(np.array(st_train.epoch), np.array(st_train.avg_perclass))
		self.l6.set_data(np.array(st_valid.epoch), np.array(st_valid.avg_perclass))
		self.axarr[1].set_xlim((1, np.uint32(st_train.epoch[-1])))
		self.axarr[1].set_ylim((0,1))
		plt.pause(.0001)

		# conf. matrix for training
		for r_gt in range(0, st_train.confm_list[-1].shape[0]):
			st_train.confm_list[-1][r_gt, :] /= (st_train.confm_list[-1][r_gt, :].sum() + 1e-5)
		plt.figure(self.fig2.number)
		self.data_conf_train = self.axarr2[0].imshow(st_train.confm_list[-1], cmap=plt.cm.jet, norm=mc.Normalize(vmin=0.0, vmax=1.0, clip=False), interpolation='nearest')
		idx = 0
		for x in xrange(self.L):
			for y in xrange(self.L):
				self.array_text_training[idx].set_text(str("{:.2f}".format(st_train.confm_list[-1][x][y])))
				idx += 1

		# conf. matrix for validation
		for r_gt in range(0, st_valid.confm_list[-1].shape[0]):
			st_valid.confm_list[-1][r_gt, :] /= (st_valid.confm_list[-1][r_gt, :].sum() + 1e-5)
		plt.figure(self.fig2.number)
		self.data_conf_train = self.axarr2[1].imshow(st_valid.confm_list[-1], cmap=plt.cm.jet, norm=mc.Normalize(vmin=0.0, vmax=1.0, clip=False), interpolation='nearest')
		idx = 0
		for x in xrange(self.L):
			for y in xrange(self.L):
				self.array_text_validation[idx].set_text(str("{:.2f}".format(st_valid.confm_list[-1][x][y])))
				idx += 1

		plt.draw()
		plt.pause(.0001)

		# random images!
		plt.figure(self.fig3.number)
		for i in range(len(self.listManagers)):
			if(self.policy == 'fixed'):
				(rgb, gt) = self.listSampleImages[i]
			else:
				(rgb, gt) = self.listManagers[i].getRandomSamples(self.num_random_imgs[i])
			
			# get predictions in GPU
			self.last_net(rgb, gt, test_mode=True)
			pred = self.last_net.probs.data.argmax(1)

			# move data back to CPU
			pred_ = cuda.to_cpu(pred)
			rgb_ = cuda.to_cpu(rgb.data).astype(np.uint8)
			gt_ = cuda.to_cpu(gt.data)

			#ipdb.set_trace()

			# labels range from -1 to C-1 in chainer, where -1 means "ignore"
			gt_ = self.LUT[gt_+1,:].squeeze()
			gt_ = gt_.transpose((0,3,1,2))

			pred_ = self.LUT[pred_+1,:].squeeze()
			pred_ = pred_.transpose((0,3,1,2))

			#ipdb.set_trace()
						

			(A, B, W, H) = rgb_.shape
			comp_image = np.ndarray((3, W*3, H*self.num_random_imgs[i]), dtype=np.uint8)	

			for j in range(self.num_random_imgs[i]):
				comp_image[:, 0:W, j*H:(j+1)*H] = rgb_[j, :, :, :] 
				comp_image[:, W:2*W, j*H:(j+1)*H] = pred_[j, :, :, :] 
				comp_image[:, 2*W:3*W,j*H:(j+1)*H] = gt_[j, :, :, :] 

			comp2  = comp_image.transpose((2,1,0))

			#ipdb.set_trace()
			self.axarr3[i].imshow(comp2)


		plt.draw()
		plt.pause(.0001)
		
	# --------------------------------:

