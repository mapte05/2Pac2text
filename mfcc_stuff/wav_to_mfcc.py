from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from os import listdir
from os.path import isfile, join
import os
import re
import time
import cPickle as pickle
import collections
import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import random



def read_files():

	reload(sys)  
	sys.setdefaultencoding('utf8') #fix pickle bug
	cwd = os.getcwd()

	wavs = cwd + "/cmu_wavs/"
	dicts = [x + "/wav" for x in listdir(wavs) if os.path.isdir(wavs + "/" + x)]
	files = []
	for i in range(len(dicts)):
		temp = [wavs + dicts[i] + "/" + f for f in listdir(wavs + "/" + dicts[i]) if f.endswith(".wav")]
		print dicts[i], len(temp)
		files.extend(temp)

	print 0
	print len(files)
	start = time.time() #timing
	all_features = [] #eventually use numpy?
	seq_lengths = []
	all_features_val = [] #eventually use numpy?
	seq_lengths_val = []
	num_files = len(files)
	val_indices = [] #for the val set
	for i,file in enumerate(files):
		(rate,sig) = wav.read(file) 
		#noisy_sig = perturb(sig)
		mfcc_feat = mfcc(sig, rate)
		# noisy_mfcc_feat = mfcc(noisy_sig, rate) #can add more features later but for now just mfcc_feat
		if random.random() < .1:
			all_features_val.append(mfcc_feat)
			seq_lengths_val.append(len(mfcc_feat))
			val_indices.append(i)
		else:
			all_features.append(mfcc_feat)
			seq_lengths.append(len(mfcc_feat))

	print len(all_features_val), len(seq_lengths_val) #sanity check
	print len(all_features), len(seq_lengths)			
	labels = cwd + "/cmu_labels/"
	dicts = [x + "/labels" for x in listdir(labels) if os.path.isdir(labels + "/" + x)]
	files = []
	for i in range(len(dicts)):
		temp = [labels + dicts[i] + "/" + f for f in listdir(labels + "/" + dicts[i]) if f.endswith(".utt")]
		print dicts[i], len(temp)
		files.extend(temp)
	print 1
	print len(files)
	all_labels = []
	all_labels_val = []
	for i,file in enumerate(files):
		label = []
		with open(file, 'r') as f:
			for line in f.readlines():
				label += line.split()
		label_str = " ".join(label)
		if len(val_indices) != 0 and val_indices[0] == i:	
			all_labels_val.append(label_str)
			val_indices.pop(0)
		else:
			all_labels.append(label_str)

	print len(all_labels_val)
	print len(all_labels)
	pickle_tuple = (all_features, all_labels, seq_lengths)
	pickle_tuple_val = (all_features_val, all_labels_val, seq_lengths_val)

	with open('cmu_all12_train.dat', 'wb') as f:
		pickle.dump(pickle_tuple, f)
	with open('cmu_all12_val.dat', 'wb') as f:
		pickle.dump(pickle_tuple_val, f)

def num_to_char(num):
	alphabet = "abcdefghijklmnopqrstuvwxyz "
	return alphabet[num]

def wsj():
	cwd = os.getcwd()

	# can use this code if cmu directory is somewhere else
	# slashes = [m.start() for m in re.finditer('/', cwd)]
	# final_slash = slashes[-1]
	# rel_dir = cwd[0:final_slash+1]
	wsj = cwd + "/wsj"
	f = open(wsj + '/labels_train.pkl', 'rb')
	label_dict = pickle.load(f)
	g = open(wsj + '/mfcc_train.pkl', 'rb')
	mfcc_dict = pickle.load(g)
	features = []
	seq_lengths = []
	labels = []
	for i, key in enumerate(label_dict):
		label = ""
		num_arr = label_dict[key]
		for num in num_arr:
			label += num_to_char(num)
		# if i < 10:
		# 	for i in range(len(mfcc_dict[key])):
		# 		print len(mfcc_dict[key][i])
		labels.append(label)
		features.append(mfcc_dict[key])
		seq_lengths.append(len(mfcc_dict[key][0]))

	pickle_tuple = (features, labels, seq_lengths)
	print (len(features), len(labels), len(seq_lengths))

	with open('wsj_train.dat', 'wb') as f:
		pickle.dump(pickle_tuple, f)


def perturb(sig):
	noise_factor = .0000001
	noise = np.random.normal(min(sig)*noise_factor, max(sig)*noise_factor, size=(len(sig)))
	ret = np.add(sig, noise)
	return ret
	# else:
	# 	max_ = max(max([x[0] for x in sig]),max([x[1] for x in sig]))
	# 	min_ = min(min([x[0] for x in sig]),min([x[1] for x in sig]))
	# 	noise = np.random.normal(min_*noise_factor, max_*noise_factor, size=(len(sig), 2))
	# ret = [str(int(x[0] + x[1][0])) for x in zip(sig, noise)]
	# ret = " ".join(ret)
	# temp = []
	# return temp.append(ret)

def load():
	cwd = os.getcwd()	
	with open(cwd + '/cmu_train.dat', 'rb') as f:
		dataset = pickle.load(f) 
		for i in range(len(dataset[0])):
		    	print(dataset[0][i])
            	print(len(dataset[0][i]))
            	print(dataset[1][i])
            	print(dataset[2][i])
            	if i > 20:
              		exit(0)
	# with open('verbose.txt', 'wb') as f:
	# 	for i, line in enumerate(dataset[1]):
	# 		f.write(str(i) + '\n')
	# 		f.write(line + '\n')

def get_frequencies():
	cwd = os.getcwd()	
	dicts = ["/cmu_us_bdl_arctic/wav", "/cmu_us_clb_arctic/wav", "/cmu_us_rms_arctic/wav", "/cmu_us_slt_arctic/wav"]
	files = [[f for f in listdir(cwd + dicts[0])], [f for f in listdir(cwd + dicts[1])], [f for f in listdir(cwd + dicts[2])], [f for f in listdir(cwd + dicts[3])]]
	for i in range(len(dicts)):
		first = True
		for j,file in enumerate(files[i]): #stolen from https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files
			# f.write(file + "\n") 
			rate, sig = wav.read(cwd + dicts[i] + "/" + file) # load the data
			print rate
			exit(0)
			normalized =[(elem/2**16.)*2 - 1 for elem in sig] # this is 8-bit track, b is now normalized on [-1,1)
			ft = fft(normalized)
			unique_pts = abs(ft[:(len(ft)/2 - 1)])
			print unique_pts[0:100]

			max_ = 0
			min_ = 0
			for elem in unique_pts:
				if elem < min_:
					min_ = elem
				if elem > max_:
					max_ = elem
			print min_, max_
			#for plotting
			# plt.plot(abs(complex_[:(len(complex_)/2-1)]),'r') # you only need half of the fft list
			# plt.savefig(file+'.png',bbox_inches='tight')
			return

def isolate_vocals():
	cwd = os.getcwd()	
	(rate,sig) = wav.read(cwd + "/rick/Rick-Astley_-_Never-Gonna-Give-You-Up-izmenennyy-ton-2.wav") #sig[0] = left, sig[1] = right
	final = []
	for i in range(len(sig)):
		combined = (sig[i][0]/2 + sig[i][1]/2)
		removed = sig[i][0] - sig[i][1]
		final.append(combined - removed)
	print final[:100]
	wav.write(cwd + "/output/rick.wav", rate, np.array(final))


def calculate_time():
	cwd = os.getcwd()	
	dicts = ["/cmu_us_bdl_arctic/wav", "/cmu_us_clb_arctic/wav", "/cmu_us_rms_arctic/wav", "/cmu_us_slt_arctic/wav"]
	files = [[f for f in listdir(cwd + dicts[0])], [f for f in listdir(cwd + dicts[1])], [f for f in listdir(cwd + dicts[2])], [f for f in listdir(cwd + dicts[3])]]
	total = 0
	for i in range(len(dicts)):
		print i
		for j,file in enumerate(files[i]):
			print min(sig), max(sig)
			return
			rate, sig = wav.read(cwd + dicts[i] + "/" + file) # sig is the number of samples / second
			total += len(sig) / rate #total in seconds
	print total
	print total/60.
	print total/(60.*60)

wsj()
#calculate_time()
# get_frequencies()
# read_files()
# isolate_vocals()
# load()