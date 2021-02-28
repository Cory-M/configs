import numpy as np
import time
import pdb
from statistics import mode
import collections

np.random.seed(777)
D = '/home/keyu/keyu/retrieval/'
DD = '/media/chundi/ssd/oneshot_data/UCF/'
L2 = True
SAMPLE = 'center_avg' # random/center_avg/center_random
QUEUE_SIM = False
VERSION = 'v2'
WEIGHTS = None
QE = None


def my_mode(data, weights=None):
	# return most frequent item in the array
	# if multiple most frequent items, return the one appears in data first 
	if weights:
		assert len(weights) == len(data)
		weighted_data = [x for i, x in enumerate(data) for _ in range(weights[i])]
		table = collections.Counter(iter(weighted_data)).most_common()
	else:
		table = collections.Counter(iter(data)).most_common()
	if not table:
		return table
	maxfreq = table[0][1]
	for i in range(1, len(table)):
		if table[i][1] != maxfreq:
			table = table[:i]
			break
	# table: most frequent (val, frequency) pairs
	if len(table) == 1:
		return table[0][0]
	freq_item = [x[0] for x in table]
	for x in data:
		if x in freq_item:
			return x

def select_center_avg_crop(feats, labels):
	print("Labels shape: {}".format(labels.shape))
	feats = feats.reshape([-1, 30, feats.shape[-1]])
	labels = labels.reshape([-1, 30])[:,0]
	center_inds = np.arange(0, 30, 3) + 1
	feats = np.mean(feats[:, center_inds, :], axis=1)
	
	assert labels.shape[0] == feats.shape[0]
	return feats, labels

def select_random_center_crop(feats, labels):
	feats = feats.reshape([-1, 30, feats.shape[-1]])
	labels = labels.reshape([-1, 30])[:, 0]
	center_inds = np.arange(0, 30, 3) + 1
	center_feats = feats[:, center_inds, :]
	inds = np.random.randint(10, size=feats.shape[0])
	feats = np.array([center_feats[i][inds[i]] for i in range(inds.shape[0])])
	return feats, labels

def select_random_clip(feats, labels):
	print("Labels shape: {}".format(labels.shape))
	feats = feats.reshape([-1, 30, feats.shape[-1]])
	labels = labels.reshape([-1, 30])
	assert labels.shape[0] == feats.shape[0]
	inds = np.random.randint(30, size=feats.shape[0])
	feats = np.array([feats[i][inds[i]] for i in range(inds.shape[0])])
	labels= np.array([labels[i][inds[i]] for i in range(inds.shape[0])])
	return feats, labels

def perform_re(postfix):
	val_feats = np.load(D + 'queries_{}.npy'.format(postfix))
	train_feats = np.load(D + 'candidates_{}.npy'.format(postfix))
	with open(D + 'queries_{}.txt'.format(postfix)) as f:
		lines = f.readlines()
		val_class = [int(line.split()[1]) for line in lines]
	with open(D + 'candidates_{}.txt'.format(postfix)) as f:
		lines = f.readlines()
		train_class = [int(line.split()[1]) for line in lines]
	val_class = np.array(val_class)
	train_class = np.array(train_class)
	if SAMPLE == 'center_avg':
		val_feats, val_class = select_center_avg_crop(val_feats, val_class)
		train_feats, train_class = select_center_avg_crop(train_feats, train_class)
	elif SAMPLE == 'center_random':
		val_feats, val_class = select_random_center_crop(val_feats, val_class)
		train_feats, train_class = select_random_center_crop(train_feats, train_class) 
	elif SAMPLE == 'random':
		val_feats, val_class = select_random_clip(val_feats, val_class)
		train_feats, train_class = select_random_clip(train_feats, train_class)
	if QUEUE_SIM:
		inds = np.random.randint(train_feats.shape[0], size=QUEUE_SIM)
		train_feats = train_feats[inds, :]
		train_class = train_class[inds]
	if L2:
		train_feats /= np.expand_dims(np.sum(np.abs(train_feats) ** 2, axis=1) ** (1. / 2), axis=1)
		val_feats /= np.expand_dims(np.sum(np.abs(val_feats) ** 2, axis=1) ** (1. / 2), axis=1)
	start = time.time()
	sims = np.matmul(val_feats, train_feats.T)
	end = time.time()
	print('Retrieval takes: {}s'.format(end - start))
	start = time.time()
	I = np.argsort(-1 * sims, axis=1)[:, :1000]
	#ranked_sims = np.array([sims[i][I[i]] for i in range(sims.shape[0])])
	end = time.time()
	print('Sorting takes: {}s'.format(end - start))
	del sims
	#np.save('{}_sims.npy'.format(postfix), ranked_sims)
	#np.save(DD + '{}_I.npy'.format(postfix), I)
	print(I.shape)

	if QE:
		topk_idx = I[:, :QE]
		topk_feats = train_feats[topk_idx]
		pdb.set_trace()
#		concat_feats = np.concatenate((val_feats, topk_feats), axis=

	k_list = [1, 5, 10, 20, 50, 100, 200, 500, 950]
	
	if VERSION == 'v1':
		pretty_print_MAP = []
		pretty_print_TOPK = []
		pretty_print_PRECISION = []
		for k in k_list:
			MAP = 0.0
			precision = 0.0
			TOPKA = 0.0
			for i in range(I.shape[0]):
				cur_correct = 0
				any_correct = 0
				AP = []
				cur_i = 0
				for j in range(k):
					cur_i += 1
					if train_class[I[i][j]] == val_class[i]:
						any_correct = 1
						cur_correct += 1
					AP.append(cur_correct / (cur_i))
				TOPKA += any_correct
				precision += (cur_correct / cur_i)
				MAP += np.mean(AP)
			pretty_print_MAP.append(MAP / I.shape[0])
			pretty_print_TOPK.append(TOPKA / I.shape[0])
			pretty_print_PRECISION.append(precision / I.shape[0])
	#		pretty_print_MAP.append(str(MAP / I.shape[0]))
	#		pretty_print_TOPK.append(str(TOPKA / I.shape[0]))
	#		print('{}: Clipwise MAP@{}: {}'.format(postfix, k, MAP / I.shape[0]))
	#		print('{}: Clipwise Precision@{}: {}'.format(postfix, k, precision / I.shape[0]))
	#		print('{}: Clipwise topk@{}: {}'.format(postfix, k, TOPKA / I.shape[0]))
	#	print(' '.join(pretty_print_TOPK))
	#	print(' '.join(pretty_print_MAP))
		for i in range(len(pretty_print_MAP)):
			print('{:.3f}\t{:.3f}\t{:.3f}'.format(pretty_print_MAP[i], 
							pretty_print_TOPK[i],
							pretty_print_PRECISION[i]))
		print('\n')

	elif VERSION == 'v2': # voting and measure by classification accuracy
		I_label = train_class[I] #(10000, 1000)
		pretty_print_ACC = []
		for k in k_list:
			print(k)
			any_correct = 0
			for i in range(I_label[:,:k].shape[0]):
				if WEIGHTS == 'linear':
					weights = range(k, 0, -1)
				else:
					weights = None
				pred = my_mode(I_label[i, :k], weights)
				if pred == val_class[i]:
					any_correct += 1
			pretty_print_ACC.append(any_correct / I.shape[0])
		print(postfix)
		for acc in pretty_print_ACC:
			print('{:.3f}'.format(acc))

#postfixs = ['fs_020_semi_wo_mining', 'fs_020only_sec_loadSelfSup']
#postfixs = ['sbn240']
postfixs = ['fs_020only_sec_loadSelfSup']#, 'fs_020only_sec_loadSelfSup_whole'] 
for p in postfixs:
	perform_re(p)