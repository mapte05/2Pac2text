# CITATION: code below relies heavily on code from cs224s homework 3 question 2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves.urllib.request import urlretrieve
from six.moves import xrange as range

import os
import pdb
import sys
import numpy as np
import os.path
import tensorflow as tf
# import cPickle as pickle
try:
   import cPickle as pickle
except:
   import pickle


import string


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.

        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths


# def get_tidigits_to_index_mapping():
# 	return {"z": 0, "o": 10, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "_": 11}


def get_chars_to_index_mapping():
    result = {}
    for i, let in enumerate(string.ascii_lowercase):
        result[let] = i
    # now add common punctuation marks
    last = len(string.ascii_lowercase)
    to_add = [".", "?", "!", "'", ","]
    for i, punc in enumerate(to_add):
        result[punc] = last + i
    # always add blank at end
    result[" "] = last + len(to_add) # assumes we don't see tabs 
    return result


def compare_predicted_to_true(preds, trues_tup):
    # inv_index_mapping = {v: k for k, v in get_tidigits_to_index_mapping().items()}     
    inv_index_mapping = {v: k for k, v in get_chars_to_index_mapping().items()}      

    preds = tf.sparse_tensor_to_dense(preds, default_value=-1).eval()
    trues = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=trues_tup[0], values=trues_tup[1], dense_shape=trues_tup[2]), default_value=-1).eval()

    for true, pred in zip(trues, preds):
        predicted_label = "".join([inv_index_mapping[ch] for ch in pred if ch != -1])
        true_label = "".join([inv_index_mapping[ch] for ch in true if ch != -1])

        print("Predicted: {}\n   Actual: {}\n".format(predicted_label, true_label))
        

def preprocess_lyrics(lyrics):
    # lyrics: list of strings corresponding to lyrics for each song/utterance
    result = []
    chars = get_chars_to_index_mapping()
    for l in lyrics:
        lower = l.lower()
        r = []
        for c in lower:
            if c in chars:
                r.append(chars[c])
            elif c == "\t":
                print("HOW DID THAT TAB GET IN THERE??")
            else:
                print("character: %s unknown" % c)
        result.append(r)
    return result


def reverse(lyrics):
    # sanity check function for preprocessing function
    inv_index_mapping = {v: k for k, v in get_chars_to_index_mapping().items()}
    result = []
    chars = inv_index_mapping
    for l in lyrics:
        lower = l
        r = []
        for c in lower:
            if c in chars:
                r.append(chars[c])
            elif c == "\t":
                print("HOW DID THAT TAB GET IN THERE??")
            else:
                print("character: %s unknown" % c)
        result.append("".join(r))
    return result



def load_dataset(dataset_path):
	with open(dataset_path, 'rb') as f:
		dataset = pickle.load(f)
    # assumes (examples, lyrics, seq len) format
    sequences = preprocess_lyrics(dataset[1])
    return (dataset[0], sequences, dataset[2])



def make_batches(dataset, batch_size=16):
    examples = []
    sequences = []
    seqlens = []

    l1, l2, l3 = dataset

    for i in range(0, len(l1), batch_size):
        examples.append(l1[i:i + batch_size])
        sequences.append(sparse_tuple_from(l2[i:i + batch_size]))
        seqlens.append(l3[i:i + batch_size])

    return examples, sequences, seqlens