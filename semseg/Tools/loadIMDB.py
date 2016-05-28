import h5py

def loadIMDB(matfile):
	M = h5py.File(matfile)

	data = M[M.keys()[1]]['images']['data'].value
	labels = M[M.keys()[1]]['images']['labels'].value
	sets = M[M.keys()[1]]['images']['set'].value
	stats = M[M.keys()[1]]['stats'].value

	imdb = {'data':data, 'labels':labels, 'set':sets, 'stats':stats}
	return imdb
