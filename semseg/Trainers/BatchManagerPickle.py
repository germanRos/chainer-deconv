from BatchManager import BatchManager
import re
import os
import numpy as np
from scipy import misc
import ipdb

class BatchManagerPickle(BatchManager):
	"""This class deals with batches from an IMDB in memory
	"""

	def initialize(self, filename):
		pass

	def getBatch_(self, indices):
		pass
