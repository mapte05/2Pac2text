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

	artic_dir = cwd + "/cmu_us_bdl_arctic/wav" #this assumes cmu directory is in current directory
	files = [f for f in listdir(artic_dir)]
	# start = time.time() #timing
	file_features = [] #eventually use numpy?
	for i,file in enumerate(files):
		(rate,sig) = wav.read(artic_dir + "/" + file) 
		mfcc_feat = mfcc(sig,rate)
		# d_mfcc_feat = delta(mfcc_feat, 2) #can add more features later but for now just mfcc_feat
		# fbank_feat = logfbank(sig,rate)
		# features.add(mfcc_feat)
		file_features.append(mfcc_feat)
		# if i % 100 == 0:
		# 	print "File number:", i
		# 	print time.time() - start
	with open('mfcc_features', 'wb') as f:
		pickle.dump(file_features, f)

read_files()