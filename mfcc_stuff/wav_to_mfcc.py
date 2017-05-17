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

def read_files():

	reload(sys)  
	sys.setdefaultencoding('utf8') #fix pickle bug

	cwd = os.getcwd()

	# can use this code if cmu directory is somewhere else
	# slashes = [m.start() for m in re.finditer('/', cwd)]
	# final_slash = slashes[-1]
	# rel_dir = cwd[0:final_slash+1]

	artic_dir = cwd + "/cmu_us_bdl_arctic/wav" #assumes cmu directory is in current directory
	files = [f for f in listdir(artic_dir)]
	start = time.time() #timing
	all_features = [] #eventually use numpy?
	seq_lengths = []
	all_features_val = [] #eventually use numpy?
	seq_lengths_val = []
	num_files = len(files)
	for i,file in enumerate(files):
		(rate,sig) = wav.read(artic_dir + "/" + file) 
		mfcc_feat = mfcc(sig,rate) #can add more features later but for now just mfcc_feat
		# d_mfcc_feat = delta(mfcc_feat, 2) 
		# fbank_feat = logfbank(sig,rate)
		# features.add(mfcc_feat)
		if i < (num_files/10):
			all_features_val.append(mfcc_feat)
			seq_lengths_val.append(len(mfcc_feat))
		else:
			all_features.append(mfcc_feat)
			seq_lengths.append(len(mfcc_feat))
		# if i % 100 == 0:
		# 	print "File number:", i
		# 	print time.time() - start

	label_dir = cwd + "/../cmu_labels/cmu_us_bdl_arctic"
	files = [f for f in listdir(label_dir)]
	all_labels = []
	all_labels_val = []
	for i,file in enumerate(files):
		label = []
		with open(label_dir + "/" + file, 'r') as f:
			for line in f.readlines():
				label += line.split()
		if i < (num_files/10):	
			all_labels_val.append(" ".join(label))
		else:
			all_labels.append(" ".join(label))
		# if i % 100 == 0:
		# 	print "File number:", i
		# 	print time.time() - start

	pickle_tuple = (all_features, all_labels, seq_lengths)
	# print (len(all_features), len(seq_lengths), len(all_labels))
	# print pickle_tuple
	pickle_tuple_val = (all_features_val, all_labels_val, seq_lengths_val)
	# print (len(all_features_val), len(seq_lengths_val), len(all_labels_val))
	with open('cmu_train.dat', 'wb') as f:
		pickle.dump(pickle_tuple, f)
	with open('cmu_val.dat', 'wb') as f:
		pickle.dump(pickle_tuple_val, f)

def load():
	cwd = os.getcwd()	
	with open(cwd + '/cmu_train.dat', 'rb') as f:
		dataset = pickle.load(f) 

read_files()
load()