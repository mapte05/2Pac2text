#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("Queen.wav")
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 2)
#fbank_feat = logfbank(sig,rate)

print [a + b for a, b in zip(mfcc_feat, d_mfcc_feat)]