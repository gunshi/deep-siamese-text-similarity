#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
from helper import InputHelper, save_plot, compute_distance
from siamese_network import SiameseLSTM
import gzip
from random import random
from amos import Conv

# Parameters
# ==================================================

tf.flags.DEFINE_integer("embedding_dim", 1000, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("training_file_path", "/home/halwai/gta_data/final/", "training folder (default: /home/halwai/gta_data/final)")
tf.flags.DEFINE_integer("max_frames", 20, "Maximum Number of frame (default: 20)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 8, "Batch Size (default: 10)")
tf.flags.DEFINE_integer("num_epochs", 25, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 5, "Save model after this many epochs (default: 100)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


inpH = InputHelper()
train_set, dev_set, sum_no_of_batches = inpH.getDataSets(FLAGS.training_file_path, FLAGS.max_frames, 10, FLAGS.batch_size)


batches=inpH.batch_iter(
    train_set[0], train_set[1], train_set[2], FLAGS.batch_size, FLAGS.num_epochs, [[104, 114, 124], (227, 227)], is_train=True)


for nn in xrange(FLAGS.num_epochs):
    print("Epoch Number: {}".format(nn))
    epoch_start_time = time.time()
    for kk in xrange(sum_no_of_batches):
        x1_batch, x2_batch, y_batch = batches.next()
        s= raw_input("Press Enter to continue...")
        print(s)
