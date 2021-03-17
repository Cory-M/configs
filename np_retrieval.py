import numpy as np
import time
import pdb
from statistics import mode
import collections
import os

np.random.seed(777)
D = '/home/keyu/keyu/retrieval/'
DD = '/media/chundi/ssd/oneshot_data/UCF/'
L2 = True
SAMPLE = 'center_avg' # random/center_avg/center_random
QUEUE_SIM = False
VERSION = 'v2'
VOTE_WEIGHTS = None
QE = None
DOMINANT = 0.7 # a float number between [0, 1], or None
WRITE_FILE = True

def my_mode(data, weights=None, dominant=0):
	# return most frequent item in the array
	# if multiple most frequent items, return the one appears in data first 
	if weights:
		assert len(weights) == len(data)
		weighted_data = [x for i, x in enumerate(data) for _ in range(weights[i])]
		table = collections.Counter(iter(weighted_data)).most_common()
		assert dominant == 0
		dominant_thresh = 0
	else:
		table = collections.Counter(iter(data)).most_common()
		dominant_thresh = len(data) * dominant
	if not table:
		return table
	maxfreq = table[0][1]
	if maxfreq < dominant_thresh:
		return -1
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
		val_meta = [line.split('.')[0]+'.mp4' for line in lines][::30]
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
	if VERSION != 'v3':
		del sims
	#np.save('{}_sims.npy'.format(postfix), ranked_sims)
	#np.save(DD + '{}_I.npy'.format(postfix), I)
	print(I.shape)

	if QE:
		start = time.time()
		topk_idx = I[:, :QE]
		topk_feats = train_feats[topk_idx]
		concat_feats = np.concatenate(
			(val_feats.reshape(I.shape[0], 1, -1), topk_feats), 
			axis=1) # (N, 1+QE, 2028)
#		qe_weights = np.arange(QE+1, 0, -1).repeat(
#					I.shape[0]).reshape(QE+1, I.shape[0]).T.reshape(-1, QE+1, 1)
#		val_feats = np.mean(concat_feats * qe_weights, axis=1)
		val_feats = np.mean(concat_feats, axis=1)
		if L2:
			val_feats /= np.expand_dims(np.sum(np.abs(val_feats) ** 2, axis=1) ** (1. / 2), axis=1)
		sims = np.matmul(val_feats, train_feats.T)
		I = np.argsort(-1 * sims, axis=1)
		end = time.time()
		print('QE takes: {:.5f}s'.format(end-start))


#	k_list = [1, 5, 10, 20, 50, 100, 200, 500, 950]
	k_list = [5, 30, 40, 50, 60, 70, 100]
	
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
		pretty_print_COUNT = []
		for k in k_list:
			any_correct = 0
			all_count = 0
			info = []
			for i in range(I_label[:,:k].shape[0]):
				if VOTE_WEIGHTS == 'linear':
					weights = range(k, 0, -1)
				else:
					weights = None
				pred = my_mode(I_label[i, :k], weights, dominant=DOMINANT)
				if pred == val_class[i]:
					any_correct += 1
				if pred != -1:
					all_count += 1
#					line = val_meta[i] + ' ' + str(val_class[i])
					line = val_meta[i] + ' ' + str(pred)
					info.append(line)
			pretty_print_COUNT.append(all_count)
			acc = any_correct / all_count if all_count != 0 else 0
			pretty_print_ACC.append(acc)
			if WRITE_FILE:
				assert all_count == len(info)
				file_name = 'results/{}_{}_{}_{:03d}_{:.3f}_{:.4f}_{}.txt'.format(
									postfix, VERSION, DOMINANT, k, 
									acc, all_count/I.shape[0], I.shape[0])
				with open(file_name,  'w') as f:
					for line in info:
						f.write(line+'\n')
		print(postfix)
		for i, acc in enumerate(pretty_print_ACC):
			print('{:.3f}\t({}/{})'.format(acc, pretty_print_COUNT[i], I.shape[0]))
	
	elif VERSION == 'v3': # threshold
		t_list = [0.99, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4]
		I = I[:,0] #(10000,)
		pretty_print_ACC = []
		pretty_print_COUNT = []

		for t in t_list:
			info = []
			any_correct = 0
			all_count = 0
			for i in range(I.shape[0]):
				if sims[i, I[i]] > t:
					all_count += 1
#					line = val_meta[i] + ' ' + str(val_class[i])
					line = val_meta[i] + ' ' + str(train_class[I[i]])
					info.append(line)
					if train_class[I[i]] == val_class[i]:
						any_correct += 1
			pretty_print_COUNT.append(all_count)
			acc = any_correct / all_count if all_count != 0 else 0
			pretty_print_ACC.append(acc)
			if WRITE_FILE:
				assert len(info) == len(all_count)
				file_name = 'results/{}_{}_{}_{}_{}.txt'.format(
									postfix, VERSION, t, acc, all_count) 
				with open(file_name, 'w') as f:
					for line in info:
						f.write(line+'\n')

		print(postfix)
		for i, acc in enumerate(pretty_print_ACC):
			print('({})\t{:.3f}\t({}/{})'.format(t_list[i], 
							acc, pretty_print_COUNT[i], I.shape[0]))
		pdb.set_trace()
		assert len(all_lines) == all_count


#postfixs = ['fs_020_semi_wo_mining', 'fs_020only_sec_loadSelfSup']
#postfixs = ['fs_030only_sec_e80_loadSelfSup']#, 'sbn240', 'fs_020only_sec_loadSelfSup_whole']
#postfixs = ['sbn240']
postfixs = ['fs_020only_sec_loadSelfSup']
for p in postfixs:
	os.system('echo '+p+'>> file.out')
	perform_re(p)
