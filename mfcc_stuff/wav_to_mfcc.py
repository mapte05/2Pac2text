from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from os import listdir
from os.path import isfile, join
import os
import re
import time
import pickle
import collections

def read_files():
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
	for i,file in enumerate(files):
		(rate,sig) = wav.read(artic_dir + "/" + file) 
		mfcc_feat = mfcc(sig,rate) #can add more features later but for now just mfcc_feat
		# d_mfcc_feat = delta(mfcc_feat, 2) 
		# fbank_feat = logfbank(sig,rate)
		# features.add(mfcc_feat)
		all_features.append(mfcc_feat)
		seq_lengths.append(len(mfcc_feat))
		# if i % 100 == 0:
		# 	print "File number:", i
		# 	print time.time() - start

	label_dir = cwd + "/../cmu_labels/cmu_us_bdl_arctic"
	files = [f for f in listdir(label_dir)]
	all_labels = []
	for i,file in enumerate(files):
		label = ""
		with open(label_dir + "/" + file, 'r') as f:
			for line in f.readlines():
				label += line.rstrip('\n')
		all_labels.append(label)
		# if i % 100 == 0:
		# 	print "File number:", i
		# 	print time.time() - start

	pickle_tuple = (all_features, seq_lengths, all_labels)
	# print pickle_tuple
	with open('mfcc_features', 'wb') as f:
		pickle.dump(pickle_tuple, f)


read_files()