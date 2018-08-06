'''
for general process usage
'''
import numpy as np

def dist(v1, v2, type='euclidean'):
	if type == 'euclidean':
		d = numpy.linalg.norm(v1-v2)
	elif type == 'hamming':
		pass
	elif type == 'geodesic':
		pass
	elif type == 'jaccard':
		pass
	else:
		d = 'null' # type can't be found

	return d