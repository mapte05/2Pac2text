# CITATION: code below relies heavily on code from cs224s homework 3 question 2

# USE THIS CODE TO AVOID RESTORATION BUG (??)

#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import math
import random
import os
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range

from utils import *
import pdb
from time import gmtime, strftime


NUM_CLASSES = len(get_chars_to_index_mapping()) # from utils, guarantees correspondence to vocabularly, includes blank

NUM_HIDDEN_LAYERS = 2


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    context_size = 0
    num_mfcc_features = 13
    num_final_features = num_mfcc_features * (2 * context_size + 1)

    batch_size = 16

    num_classes = NUM_CLASSES
    num_hidden = 100

    num_epochs = 50
    l2_lambda = 0.01
    lr = 1e-3


class CTCModel():
    """
    Implements a recursive neural network with a single hidden layer attached to CTC loss.
    This network will predict a sequence of TIDIGITS (e.g. z1039) for a given audio wav file.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph:

        inputs_placeholder: Input placeholder tensor of shape (None, None, num_final_features), type tf.float32
        targets_placeholder: Sparse placeholder, type tf.int32. You don't need to specify shape dimension.
        seq_lens_placeholder: Sequence length placeholder tensor of shape (None), type tf.int32

        TODO: Add these placeholders to self as the instance variables
            self.inputs_placeholder
            self.targets_placeholder
            self.seq_lens_placeholder

        HINTS:
            - Use tf.sparse_placeholder(tf.int32) for targets_placeholder. This is required by TF's ctc_loss op. 
            - Inputs is of shape [batch_size, max_timesteps, num_final_features], but we allow flexible sizes for
              batch_size and max_timesteps (hence the shape definition as [None, None, num_final_features]. 

        (Don't change the variable names)
        """
        inputs_placeholder = None
        targets_placeholder = None
        seq_lens_placeholder = None

        inputs_placeholder = tf.placeholder(tf.float32, shape=(None, None, Config.num_final_features))
        targets_placeholder = tf.sparse_placeholder(tf.int32)
        seq_lens_placeholder = tf.placeholder(tf.int32, shape=(None))

        self.inputs_placeholder = inputs_placeholder
        self.targets_placeholder = targets_placeholder
        self.seq_lens_placeholder = seq_lens_placeholder


    def create_feed_dict(self, inputs_batch, targets_batch, seq_lens_batch):
        """Creates the feed_dict for the digit recognizer.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.

        Args:
            inputs_batch:  A batch of input data.
            targets_batch: A batch of targets data.
            seq_lens_batch: A batch of seq_lens data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
                self.inputs_placeholder: inputs_batch,
                self.targets_placeholder: targets_batch,
                self.seq_lens_placeholder: seq_lens_batch
        }

        return feed_dict

    def add_prediction_op(self):
        """Applies a GRU RNN over the input data, then an affine layer projection. Steps to complete 
        in this function: 

        - Roll over inputs_placeholder with GRUCell, producing a Tensor of shape [batch_s, max_timestep,
          num_hidden]. 
        - Apply a W * f + b transformation over the data, where f is each hidden layer feature. This 
          should produce a Tensor of shape [batch_s, max_timesteps, num_classes]. Set this result to 
          "logits". 

        Remember:
            * Use the xavier initialization for matrices (W, but not b).
            * W should be shape [num_hidden, num_classes]. num_classes for our dataset is 12
            * tf.contrib.rnn.GRUCell, tf.contrib.rnn.MultiRNNCell and tf.nn.dynamic_rnn are of interest
        """
        logits = None 
        forward_cell_multi = []
        backward_cell_multi = []
        for _ in range(NUM_HIDDEN_LAYERS):
            forward_cell = tf.contrib.rnn.GRUCell(Config.num_hidden, activation=tf.nn.relu)
            forward_cell_multi.append(forward_cell)
            backward_cell = tf.contrib.rnn.GRUCell(Config.num_hidden, activation=tf.nn.relu)
            backward_cell_multi.append(backward_cell)

        forward_cell_multi = tf.contrib.rnn.MultiRNNCell(forward_cell_multi)
        backward_cell_multi = tf.contrib.rnn.MultiRNNCell(backward_cell_multi)
        tuple_layer_outputs, _ = tf.nn.bidirectional_dynamic_rnn(forward_cell_multi, backward_cell_multi, self.inputs_placeholder, dtype=tf.float32)
        outputs = tf.concat(tuple_layer_outputs, 2)
        W = tf.get_variable(name="W", shape=[Config.num_hidden * 2, Config.num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="b", shape=(Config.num_classes,), dtype=tf.float32, initializer=tf.zeros_initializer())
        max_timesteps = tf.shape(outputs)[1]
        num_hidden = tf.shape(outputs)[2]
        f = tf.reshape(outputs, [-1, num_hidden])
        logits = tf.matmul(f, W) + b
        logits = tf.reshape(logits, [-1, max_timesteps, Config.num_classes])

        self.logits = logits


    def add_loss_op(self):
        """Adds Ops for the loss function to the computational graph. 

        - Use tf.nn.ctc_loss to calculate the CTC loss for each example in the batch. You'll need self.logits,
          self.targets_placeholder, self.seq_lens_placeholder for this. Set variable ctc_loss to
          the output of tf.nn.ctc_loss
        - You will need to first tf.transpose the data so that self.logits is shaped [max_timesteps, batch_s, 
          num_classes]. 
        - Configure tf.nn.ctc_loss so that identical consecutive labels are allowed
        - Compute L2 regularization cost for all trainable variables. Use tf.nn.l2_loss(var). 

        """
        ctc_loss = []
        l2_cost = 0.0

        self.logits = tf.transpose(self.logits, perm=[1, 0, 2])
        ctc_loss = tf.nn.ctc_loss(self.targets_placeholder, self.logits, self.seq_lens_placeholder, ctc_merge_repeated=False, preprocess_collapse_repeated=False)
        for var in tf.trainable_variables():
            if len(var.get_shape().as_list()) > 1:
                l2_cost += tf.nn.l2_loss(var)

        # Remove inf cost training examples (no path found, yet)
        loss_without_invalid_paths = tf.boolean_mask(ctc_loss, tf.less(ctc_loss, tf.constant(10000.)))
        self.num_valid_examples = tf.cast(tf.shape(loss_without_invalid_paths)[0], tf.int32)
        cost = tf.reduce_mean(loss_without_invalid_paths) 

        self.loss = Config.l2_lambda * l2_cost + cost               

    def add_training_op(self):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables. The Op returned by this
        function is what must be passed to the `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model. Call optimizer.minimize() on self.loss. 

        """
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.lr).minimize(self.loss)
        self.optimizer = optimizer

    def add_decoder_and_cer_op(self):
        """Setup the decoder and add the word error rate calculations here. 

        Tip: You will find tf.nn.ctc_beam_search_decoder and tf.edit_distance methods useful here. 
        Also, report the mean cer over the batch in variable cer

        """        
        decoded_sequence = None 
        cer = None 

        decoded_sequence = tf.nn.ctc_beam_search_decoder(self.logits, self.seq_lens_placeholder, merge_repeated=False)[0][0]
        cer = tf.reduce_mean(tf.edit_distance(tf.to_int32(decoded_sequence), self.targets_placeholder))

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("cer", cer)

        self.decoded_sequence = decoded_sequence
        self.cer = cer

    def add_summary_op(self):
        self.merged_summary_op = tf.summary.merge_all()


    # This actually builds the computational graph 
    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_training_op()       
        self.add_decoder_and_cer_op()
        self.add_summary_op()
        

    def train_on_batch(self, session, train_inputs_batch, train_targets_batch, train_seq_len_batch, train=True):
        feed = self.create_feed_dict(train_inputs_batch, train_targets_batch, train_seq_len_batch)
        batch_cost, cer, batch_num_valid_ex, summary = session.run([self.loss, self.cer, self.num_valid_examples, self.merged_summary_op], feed)

        if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
            return 0
        if train:
            _ = session.run([self.optimizer], feed)
        return batch_cost, cer, summary

    def print_results(self, session, train_inputs_batch, train_targets_batch, train_seq_len_batch):
        train_feed = self.create_feed_dict(train_inputs_batch, train_targets_batch, train_seq_len_batch)
        train_first_batch_preds = session.run(self.decoded_sequence, feed_dict=train_feed)
        compare_predicted_to_true(train_first_batch_preds, train_targets_batch)        

    def __init__(self):
        self.build()

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', nargs='?', default='./mfcc_stuff/cmu_train.dat', type=str, help="Give path to training data")
    parser.add_argument('--val_path', nargs='?', default='./mfcc_stuff/cmu_val.dat', type=str, help="Give path to val data")
    parser.add_argument('--save_every', nargs='?', default=None, type=int, help="Save model every x iterations. Default is not saving at all.")
    parser.add_argument('--print_every', nargs='?', default=10, type=int, help="Print some training and val examples (true and predicted sequences) every x iterations. Default is 10")
    parser.add_argument('--save_to_file', nargs='?', default='saved_models/saved_model_epoch', type=str, help="Provide filename prefix for saving intermediate models")
    parser.add_argument('--load_from_file', nargs='?', default=None, type=str, help="Provide filename to load saved model")
    parser.add_argument('--test_path', nargs='?', default="no", type=str, help="Provide test filename to do test classification")
    return parser.parse_args()

def train_model(logs_path, num_batches_per_epoch, 
        train_feature_minibatches, train_labels_minibatches, train_seqlens_minibatches,
        val_feature_minibatches, val_labels_minibatches, val_seqlens_minibatches):
    with tf.Graph().as_default():
        # model = CTCModel()

        # Begin CTC Model creation (instead of above line)

        inputs_placeholder = tf.placeholder(tf.float32, shape=(None, None, Config.num_final_features))
        targets_placeholder = tf.sparse_placeholder(tf.int32)
        seq_lens_placeholder = tf.placeholder(tf.int32, shape=(None))

        forward_cell_multi = []
        backward_cell_multi = []
        for _ in range(NUM_HIDDEN_LAYERS):
            forward_cell = tf.contrib.rnn.GRUCell(Config.num_hidden, activation=tf.nn.relu)
            forward_cell_multi.append(forward_cell)
            backward_cell = tf.contrib.rnn.GRUCell(Config.num_hidden, activation=tf.nn.relu)
            backward_cell_multi.append(backward_cell)

        forward_cell_multi = tf.contrib.rnn.MultiRNNCell(forward_cell_multi)
        backward_cell_multi = tf.contrib.rnn.MultiRNNCell(backward_cell_multi)
        tuple_layer_outputs, _ = tf.nn.bidirectional_dynamic_rnn(forward_cell_multi, backward_cell_multi, self.inputs_placeholder, dtype=tf.float32)
        outputs = tf.concat(tuple_layer_outputs, 2)
        W = tf.get_variable(name="W", shape=[Config.num_hidden * 2, Config.num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="b", shape=(Config.num_classes,), dtype=tf.float32, initializer=tf.zeros_initializer())
        max_timesteps = tf.shape(outputs)[1]
        num_hidden = tf.shape(outputs)[2]
        f = tf.reshape(outputs, [-1, num_hidden])
        logits = tf.matmul(f, W) + b
        logits = tf.reshape(logits, [-1, max_timesteps, Config.num_classes])

        ctc_loss = []
        l2_cost = 0.0

        logits = tf.transpose(logits, perm=[1, 0, 2])
        ctc_loss = tf.nn.ctc_loss(targets_placeholder, logits, seq_lens_placeholder, ctc_merge_repeated=False, preprocess_collapse_repeated=False)
        for var in tf.trainable_variables():
            if len(var.get_shape().as_list()) > 1:
                l2_cost += tf.nn.l2_loss(var)

        # Remove inf cost training examples (no path found, yet)
        loss_without_invalid_paths = tf.boolean_mask(ctc_loss, tf.less(ctc_loss, tf.constant(10000.)))
        num_valid_examples = tf.cast(tf.shape(loss_without_invalid_paths)[0], tf.int32)
        cost = tf.reduce_mean(loss_without_invalid_paths) 

        loss = Config.l2_lambda * l2_cost + cost

        optimizer = tf.train.AdamOptimizer(learning_rate=Config.lr).minimize(loss)

        decoded_sequence = tf.nn.ctc_beam_search_decoder(logits, seq_lens_placeholder, merge_repeated=False)[0][0]
        cer = tf.reduce_mean(tf.edit_distance(tf.to_int32(decoded_sequence), targets_placeholder))

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("cer", cer)

        merged_summary_op = tf.summary.merge_all()

        # End CTC Model creation

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.trainable_variables())
        with tf.Session() as session:
            # Initializate the weights and biases
            session.run(init)
            if args.load_from_file is not None:
                new_saver = tf.train.import_meta_graph('%s.meta'%args.load_from_file, clear_devices=True)
                new_saver.restore(session, args.load_from_file)
                print("model restored with the %s checkpoint" % args.load_from_file)
            train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)
            val_writer = tf.summary.FileWriter(logs_path + '/val', session.graph)

            global_start = time.time()
            step_ii = 0
            #for curr_epoch in range(Config.num_epochs):
            curr_epoch = 0
            while True: # make this run forever on our google compute cpu for convergence
                total_train_cost = total_train_cer = 0
                start = time.time()
                for batch in random.sample(range(num_batches_per_epoch),num_batches_per_epoch):
                    cur_batch_size = len(train_seqlens_minibatches[batch])

                    # Train on batch function
                    train = True

                    feed = {
                            inputs_placeholder: train_feature_minibatches[batch],
                            targets_placeholder: train_labels_minibatches[batch],
                            seq_lens_placeholder: train_seqlens_minibatches[batch]
                    }
                    batch_cost, cer, batch_num_valid_ex, summary = session.run([loss, cer, num_valid_examples, merged_summary_op], feed)

                    if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
                        return 0
                    _ = session.run([optimizer], feed)
                    # batch_cost, batch_cer, summary = model.train_on_batch(session, train_feature_minibatches[batch], train_labels_minibatches[batch], train_seqlens_minibatches[batch], train=True)
                    
                    total_train_cost += batch_cost * cur_batch_size
                    total_train_cer += batch_cer * cur_batch_size
                    train_writer.add_summary(summary, step_ii)
                    step_ii += 1 
                train_cost = total_train_cost / num_examples
                train_cer = total_train_cer / num_examples

                # Train on batch function (with train set to false)
                train = False

                feed = {
                        inputs_placeholder: val_feature_minibatches[0],
                        targets_placeholder: val_labels_minibatches[0],
                        seq_lens_placeholder: val_seqlens_minibatches[0]
                }
                val_batch_cost, val_batch_cer, summary = session.run([loss, cer, num_valid_examples, merged_summary_op], feed)

                if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
                    return 0

                val_writer.add_summary(summary, step_ii)

                # val_batch_cost, val_batch_cer, _ = model.train_on_batch(session, val_feature_minibatches[0], val_labels_minibatches[0], val_seqlens_minibatches[0], train=False)
                
                log = "Epoch {}/{}, train_cost = {:.3f}, train_ed = {:.3f}, val_cost = {:.3f}, val_ed = {:.3f}, time = {:.3f}"
                print(log.format(curr_epoch+1, Config.num_epochs, train_cost, train_cer, val_batch_cost, val_batch_cer, time.time() - start))

                all_vals = session.run(tf.trainable_variables())
                weights_sum = 0
                for val in all_vals:
                    weights_sum += np.sum(val)
                print("Total sum of weights: ", weights_sum)
                print(len(all_vals))

                if args.print_every is not None and (curr_epoch + 1) % args.print_every == 0: 
                    batch_ii = 0
                    # model.print_results(session, train_feature_minibatches[batch_ii], train_labels_minibatches[batch_ii], train_seqlens_minibatches[batch_ii])

                    # Print results function
                    train_feed = {
                            inputs_placeholder: train_feature_minibatches[batch_ii],
                            targets_placeholder: train_labels_minibatches[batch_ii],
                            seq_lens_placeholder: train_seqlens_minibatches[batch_ii]
                    }

                    train_first_batch_preds = session.run(decoded_sequence, feed_dict=train_feed)
                    compare_predicted_to_true(train_first_batch_preds, train_labels_minibatches[batch_ii])   

                if args.save_every is not None and args.save_to_file is not None and (curr_epoch + 1) % args.save_every == 0:
                    saver.save(session, args.save_to_file, global_step=curr_epoch + 1)
                curr_epoch += 1

def test(test_dataset, trained_weights_file):
    test_feature_minibatches, test_labels_minibatches, test_seqlens_minibatches = make_batches(test_dataset, batch_size=len(test_dataset[0]))
    test_feature_minibatches = pad_all_batches(test_feature_minibatches)

    # FOR SANITY CHECKING
    val_dataset = load_dataset(args.val_path)
    val_feature_minibatches, val_labels_minibatches, val_seqlens_minibatches = make_batches(val_dataset, batch_size=len(val_dataset[0]))
    val_feature_minibatches = pad_all_batches(val_feature_minibatches)
    # END SANITY CHECK CODE

    with tf.Graph().as_default():
        # model = CTCModel()

        # Begin CTC Model creation (instead of above line)

        inputs_placeholder = tf.placeholder(tf.float32, shape=(None, None, Config.num_final_features))
        targets_placeholder = tf.sparse_placeholder(tf.int32)
        seq_lens_placeholder = tf.placeholder(tf.int32, shape=(None))

        forward_cell_multi = []
        backward_cell_multi = []
        for _ in range(NUM_HIDDEN_LAYERS):
            forward_cell = tf.contrib.rnn.GRUCell(Config.num_hidden, activation=tf.nn.relu)
            forward_cell_multi.append(forward_cell)
            backward_cell = tf.contrib.rnn.GRUCell(Config.num_hidden, activation=tf.nn.relu)
            backward_cell_multi.append(backward_cell)

        forward_cell_multi = tf.contrib.rnn.MultiRNNCell(forward_cell_multi)
        backward_cell_multi = tf.contrib.rnn.MultiRNNCell(backward_cell_multi)
        tuple_layer_outputs, _ = tf.nn.bidirectional_dynamic_rnn(forward_cell_multi, backward_cell_multi, self.inputs_placeholder, dtype=tf.float32)
        outputs = tf.concat(tuple_layer_outputs, 2)
        W = tf.get_variable(name="W", shape=[Config.num_hidden * 2, Config.num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="b", shape=(Config.num_classes,), dtype=tf.float32, initializer=tf.zeros_initializer())
        max_timesteps = tf.shape(outputs)[1]
        num_hidden = tf.shape(outputs)[2]
        f = tf.reshape(outputs, [-1, num_hidden])
        logits = tf.matmul(f, W) + b
        logits = tf.reshape(logits, [-1, max_timesteps, Config.num_classes])

        ctc_loss = []
        l2_cost = 0.0

        logits = tf.transpose(logits, perm=[1, 0, 2])
        ctc_loss = tf.nn.ctc_loss(targets_placeholder, logits, seq_lens_placeholder, ctc_merge_repeated=False, preprocess_collapse_repeated=False)
        for var in tf.trainable_variables():
            if len(var.get_shape().as_list()) > 1:
                l2_cost += tf.nn.l2_loss(var)

        # Remove inf cost training examples (no path found, yet)
        loss_without_invalid_paths = tf.boolean_mask(ctc_loss, tf.less(ctc_loss, tf.constant(10000.)))
        num_valid_examples = tf.cast(tf.shape(loss_without_invalid_paths)[0], tf.int32)
        cost = tf.reduce_mean(loss_without_invalid_paths) 

        loss = Config.l2_lambda * l2_cost + cost

        optimizer = tf.train.AdamOptimizer(learning_rate=Config.lr).minimize(loss)

        decoded_sequence = tf.nn.ctc_beam_search_decoder(logits, seq_lens_placeholder, merge_repeated=False)[0][0]
        cer = tf.reduce_mean(tf.edit_distance(tf.to_int32(decoded_sequence), targets_placeholder))

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("cer", cer)

        merged_summary_op = tf.summary.merge_all()

        # End CTC Model creation

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            all_vals = session.run(tf.trainable_variables())
            weights_sum = 0
            for val in all_vals:
                weights_sum += np.sum(val)
            print("Total sum of weights: ", weights_sum)
            print(len(all_vals))

            new_saver = tf.train.import_meta_graph('%s.meta'%args.load_from_file, clear_devices=True)
            new_saver.restore(session, trained_weights_file)
            print("model restored with the %s checkpoint" % trained_weights_file)

            all_vals = session.run(tf.trainable_variables())
            weights_sum = 0
            for val in all_vals:
                weights_sum += np.sum(val)
            print("Total sum of weights: ", weights_sum)
            print(len(all_vals))

            # now begin testing
            start = time.time()
            # test_batch_cost, test_batch_cer, _ = model.train_on_batch(session, test_feature_minibatches[0], test_labels_minibatches[0], test_seqlens_minibatches[0], train=False)

            # Train on batch code

            # Train on batch function (with train set to false)
            train = False

            feed = {
                    inputs_placeholder: test_feature_minibatches[0],
                    targets_placeholder: test_labels_minibatches[0],
                    seq_lens_placeholder: test_seqlens_minibatches[0]
            }
            test_batch_cost, test_batch_cer, _ = session.run([loss, cer, num_valid_examples, merged_summary_op], feed)

            if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
                return 0


            # FOR SANITY CHECKING
            # This should print out the same thing as the val_cost and val_ed for the saved run.
            val_batch_cost, val_batch_cer, _ = model.train_on_batch(session, val_feature_minibatches[0], val_labels_minibatches[0], val_seqlens_minibatches[0], train=False)
            # Train on batch function (with train set to false)
            train = False

            feed = {
                    inputs_placeholder: val_feature_minibatches[0],
                    targets_placeholder: val_labels_minibatches[0],
                    seq_lens_placeholder: val_seqlens_minibatches[0]
            }
            val_batch_cost, val_batch_cer, _ = session.run([loss, cer, num_valid_examples, merged_summary_op], feed)

            if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
                return 0

            log = "val_cost = {:.3f}, val_ed = {:.3f}, time = {:.3f}"
            print(log.format(val_batch_cost, val_batch_cer, time.time() - start))
            # END SANITY CHECK CODE

            log = "test_cost = {:.3f}, test_ed = {:.3f}, time = {:.3f}"
            print(log.format(test_batch_cost, test_batch_cer, time.time() - start))
            # if args.print_every is not None: 
            #     batch_ii = 0
            #     model.print_results(session, test_feature_minibatches[batch_ii], test_labels_minibatches[batch_ii], test_seqlens_minibatches[batch_ii])

            #     # Print results function
            #     test_feed = {
            #             inputs_placeholder: test_feature_minibatches[batch_ii],
            #             targets_placeholder: test_labels_minibatches[batch_ii],
            #             seq_lens_placeholder: test_seqlens_minibatches[batch_ii]
            #     }

            #     test_first_batch_preds = session.run(decoded_sequence, feed_dict=test_feed)
            #     compare_predicted_to_true(test_first_batch_preds, test_labels_minibatches[batch_ii])   


def pad_all_batches(batch_feature_array):
    for batch_num in range(len(batch_feature_array)):
        batch_feature_array[batch_num] = pad_sequences(batch_feature_array[batch_num])[0]
    return batch_feature_array

if __name__ == "__main__":
    args = load_args()
    if args.test_path=="no":
        logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime()) + ("_lr=%f_l2=%f" % (Config.lr, Config.l2_lambda))
        train_dataset = load_dataset(args.train_path)
        # uncomment to overfit data set
        train_dataset = (train_dataset[0][:10], train_dataset[1][:10], train_dataset[2][:10])

        train_feature_minibatches, train_labels_minibatches, train_seqlens_minibatches = make_batches(train_dataset, batch_size=Config.batch_size)
        val_dataset = load_dataset(args.val_path)
        val_feature_minibatches, val_labels_minibatches, val_seqlens_minibatches = make_batches(val_dataset, batch_size=len(val_dataset[0]))

        train_feature_minibatches = pad_all_batches(train_feature_minibatches)
        val_feature_minibatches = pad_all_batches(val_feature_minibatches)
        num_examples = np.sum([batch.shape[0] for batch in train_feature_minibatches])
        num_batches_per_epoch = int(math.ceil(num_examples / Config.batch_size))

        train_model(logs_path, num_batches_per_epoch, 
            train_feature_minibatches, train_labels_minibatches, train_seqlens_minibatches,
            val_feature_minibatches, val_labels_minibatches, val_seqlens_minibatches)
    else: # means we are testing!
        if args.load_from_file is None:
            raise ValueError("must specify weights to load through --load_from_file")
        test_dataset = load_dataset(args.test_path)
        test(test_dataset, args.load_from_file)
    