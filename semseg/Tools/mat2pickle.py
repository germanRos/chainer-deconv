import h5py
import pickle as pickle

def mat2pickle(matfile, outfile):
	M = h5py.File(matfile)

	#data = M['imdb_dense_all_reducedClasses']['images']['data'].value
	labels = M['imdb_dense_all_reducedClasses']['images']['labels'].value
	sets = M['imdb_dense_all_reducedClasses']['images']['set'].value
	stats = M['imdb_dense_all_reducedClasses']['stats'].value

	outDic = {'data':[], 'labels':labels, 'set':sets, 'stats':stats}

	fd = open(outfile, 'wb')
	pickle.dump(outDic, fd, 3)
	fd.close()


## example of use
mat2pickle('/home/gros/imdb_dense_all_reducedClasses.mat', '/home/gros/imdb_test.pickle')
