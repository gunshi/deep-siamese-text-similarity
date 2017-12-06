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
from posenet import net
from posenet import Conv as Conv_pose
from amos import Conv as Conv_weather

# Parameters
# ==================================================

tf.flags.DEFINE_integer("embedding_dim", 1000, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularizaion lambda (default: 0.001)")
tf.flags.DEFINE_string("training_file_path", "/home/tushar/Heavy_dataset/gta_data/final/", "training folder (default: /home/halwai/gta_data/final)")
tf.flags.DEFINE_string("training_files_path", "./annotation_files/", "training folder (default: /home/halwai/gta_data/final)")
tf.flags.DEFINE_integer("max_frames", 20, "Maximum Number of frame (default: 20)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 4, "Batch Size (default: 10)")
tf.flags.DEFINE_integer("num_epochs", 15, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many epochs (default: 100)")
tf.flags.DEFINE_integer("num_lstm_layers", 3, "Number of LSTM layers(default: 1)")
tf.flags.DEFINE_integer("hidden_dim", 150, "Number of LSTM layers(default: 2)")
tf.flags.DEFINE_string("loss", "contrastive", "Type of Loss functions:: contrastive/AAAI(default: contrastive)")
tf.flags.DEFINE_boolean("projection", False, "Project Conv Layers Output to a Lower Dimensional Embedding (Default: True)")
tf.flags.DEFINE_boolean("conv_net_training", False, "Training ConvNet (Default: False)")
tf.flags.DEFINE_float("lr", 0.00001, "learning-rate(default: 0.00001)")
tf.flags.DEFINE_string("name", "result", "prefix names of the output files(default: result)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", False, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("return_outputs", 1, "Outpust from LSTM, 0=>Last LSMT output, 2=> Cell-State Output. 1=> Hidden-State Output (default: 2)")
tf.flags.DEFINE_string("summaries_dir", "/home/tushar/codes/rnn-cnn/summaries/", "Summary storage")

#Conv Net Parameters
tf.flags.DEFINE_string("conv_layer_pose", "cls3_fc1_pose", "CNN features from AMOSNet(default: cls3_fc1_pose)")
tf.flags.DEFINE_string("conv_layer_weight_pretrained_path_pose", "/home/tushar/codes/rnn-cnn/PoseNet.ckpt", "AMOSNet pre-trained weights path")#or posenet.ckpt?
tf.flags.DEFINE_string("conv_layer_weather", "pool6", "CNN features from AMOSNet(default: cls3_fc1_pose)")
tf.flags.DEFINE_string("conv_layer_weight_pretrained_path_weather", "/home/tushar/Heavy_dataset/amos/data1.npy", "AMOSNet pre-trained weights path")

tf.flags.DEFINE_string("train_file_positive", "./annotation_files/positives-moredata-train+val.txt", "Positive_training_file")
tf.flags.DEFINE_string("train_file_negative", "./annotation_files/negs-concat.txt", "Negative_training_file")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.training_files_path==None:
    print("Input Files List is empty. use --training_files_path argument.")
    exit()

inpH = InputHelper()
train_set, dev_set, sum_no_of_batches,num_pos,num_neg = inpH.getDataSets(FLAGS.training_file_path,FLAGS.training_files_path, FLAGS.max_frames,12,26 , FLAGS.batch_size, FLAGS.train_file_positive,FLAGS.train_file_negative)

# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options,
      )
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        convModel_pose = Conv_pose(
         FLAGS.conv_layer_pose,
         FLAGS.conv_layer_weight_pretrained_path_pose,
         FLAGS.batch_size,
         FLAGS.max_frames,
         FLAGS.conv_net_training)

        convNet_pose_tvar = tf.global_variables()

        convModel_weather = Conv_weather(
         FLAGS.conv_layer_weather,
         FLAGS.conv_layer_weight_pretrained_path_weather,
         FLAGS.batch_size,
         FLAGS.max_frames,
         FLAGS.conv_net_training)

        convNet_weather_tvar = list(set(tf.global_variables())- set(convNet_pose_tvar))

        siameseModel = SiameseLSTM(
            sequence_length=FLAGS.max_frames,
            input_size=2048+9216,
            embedding_size=FLAGS.embedding_dim,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            batch_size=FLAGS.batch_size,
            num_lstm_layers=FLAGS.num_lstm_layers,
            hidden_unit_dim=FLAGS.hidden_dim,
            loss=FLAGS.loss,
            projection=FLAGS.projection,
            return_outputs=FLAGS.return_outputs,
            num_pos=num_pos,
            num_neg=num_neg)

        lstm_tvar = list(set(tf.global_variables())- set(convNet_pose_tvar) - set(convNet_weather_tvar))

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate=tf.train.exponential_decay(1e-5, global_step, sum_no_of_batches*5, 0.95, staircase=False, name=None)
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        print("initialized convmodel and siamesemodel object")

    if FLAGS.conv_net_training:
        tv =  lstm_tvar+convNet_pose_tvar+convNet_weather_tvar
    else:
        tv = lstm_tvar

    regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv if 'bias' not in v.name ])
    total_loss=siameseModel.loss+FLAGS.l2_reg_lambda*regularization_cost
    grads_and_vars=optimizer.compute_gradients(total_loss, var_list = lstm_tvar)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print("defined training_ops")
    # keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    #grad_summaries_merged = tf.summary.merge(grad_summaries)
    summaries_merged = tf.summary.merge_all()
    print("defined gradient summaries")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join("/home/tushar/codes/rnn-cnn/", "runs", FLAGS.name))
    print("Writing to {}\n".format(out_dir))

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    #lstm_checkpoint_prefix = os.path.join(checkpoint_dir, "lstm_model")
    if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    lstm_savepath="/home/tushar/codes/rnn-cnn/data/lstm_outputs"
    if not os.path.exists(lstm_savepath):
            os.makedirs(lstm_savepath)

    #print all convNet_trainable parameters
    #for i, var in enumerate(convNet_tvar):
    #    print("{}".format(var.name))

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
    saver_posenet = tf.train.Saver(var_list=convNet_pose_tvar)

    #lstm_saver = tf.train.Saver([out1,out2], max_to_keep=2)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    #Fix weights for Conv Layers
    convModel_pose.initalize(sess, saver_posenet)
    convModel_weather.initalize(sess)

    #print all trainable parameters
    tvar = tv
    for i, var in enumerate(tvar):
        print("{}={}".format(i,var.name))

    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)


    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', graph=tf.get_default_graph())
    val_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/val' , graph=tf.get_default_graph())

    def train_step(x1_batch, x2_batch, y_batch, video_lengths):

        #A single training step

        [x1_batch_pose] = sess.run([convModel_pose.features],  feed_dict={convModel_pose.input_imgs: x1_batch[:,0:224,0:224,:]})
        [x2_batch_pose] = sess.run([convModel_pose.features],  feed_dict={convModel_pose.input_imgs: x2_batch[:,0:224,0:224,:]})

        [x1_batch_weather] = sess.run([convModel_weather.features],  feed_dict={convModel_weather.input_imgs: x1_batch})
        [x2_batch_weather] = sess.run([convModel_weather.features],  feed_dict={convModel_weather.input_imgs: x2_batch})

        x1_batch = np.concatenate((x1_batch_pose,x1_batch_weather),axis=1)
        x2_batch = np.concatenate((x2_batch_pose,x2_batch_weather),axis=1)

        feed_dict = {
                         siameseModel.input_x1: x1_batch,
                         siameseModel.input_x2: x2_batch,
                         siameseModel.input_y: y_batch,
                         siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
                         siameseModel.video_lengths: video_lengths,
        }

        out1, out2, _, step, loss, dist, summary = sess.run([siameseModel.out1, siameseModel.out2, tr_op_set, global_step, siameseModel.loss, siameseModel.distance, summaries_merged],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        d=compute_distance(dist, FLAGS.loss)
        correct = y_batch==d
        #print(out1, out2)
        #print(video_lengths)
        #print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, correct))
        print(y_batch, dist, d)
        return summary, np.sum(correct), loss

    def dev_step(x1_batch, x2_batch, y_batch, video_lengths, dev_iter, epoch):

        #A single training step

        [x1_batch_pose] = sess.run([convModel_pose.features],  feed_dict={convModel_pose.input_imgs: x1_batch[:,0:224,0:224,:]})
        [x2_batch_pose] = sess.run([convModel_pose.features],  feed_dict={convModel_pose.input_imgs: x2_batch[:,0:224,0:224,:]})

        [x1_batch_weather] = sess.run([convModel_weather.features],  feed_dict={convModel_weather.input_imgs: x1_batch})
        [x2_batch_weather] = sess.run([convModel_weather.features],  feed_dict={convModel_weather.input_imgs: x2_batch})

        x1_batch = np.concatenate((x1_batch_pose,x1_batch_weather),axis=1)
        x2_batch = np.concatenate((x2_batch_pose,x2_batch_weather),axis=1)

        feed_dict = {
                         siameseModel.input_x1: x1_batch,
                         siameseModel.input_x2: x2_batch,
                         siameseModel.input_y: y_batch,
                         siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
                         siameseModel.video_lengths: video_lengths,

        }

        step, loss, dist, summary, out1, out2 = sess.run([global_step, siameseModel.loss, siameseModel.distance, summaries_merged,siameseModel.out1,siameseModel.out2],  feed_dict)
        #np.save(lstm_savepath+'/out1_'+str(dev_iter)+'_'+str(epoch),out1)
        #np.save(lstm_savepath+'/out2_'+str(dev_iter)+'_'+str(epoch),out2)
        #np.save(lstm_savepath+'/y_'+str(dev_iter)+'_'+str(epoch),y_batch)
        time_str = datetime.datetime.now().isoformat()
        d=compute_distance(dist, FLAGS.loss)
        correct = y_batch==d
        #print("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, correct))
        #print(y_batch, dist, d)
        return summary, np.sum(correct), loss, correct

    # Generate batches
    batches=inpH.batch_iter(
                train_set[0], train_set[1], train_set[2], train_set[3], FLAGS.batch_size, FLAGS.num_epochs, convModel_weather.spec, shuffle=True, is_train=False)

    ptr=0
    max_validation_correct=0.0
    start_time = time.time()
    train_accuracy, val_accuracy, pos_val_accuracy, neg_val_accuracy = [] , [], [], []
    train_loss, val_loss = [], []
    train_batch_loss_arr, val_batch_loss_arr = [], []


    for nn in xrange(FLAGS.num_epochs):

        current_step = tf.train.global_step(sess, global_step)
        print("Epoch Number: {}".format(nn))
        epoch_start_time = time.time()
        sum_train_correct=0.0
        train_epoch_loss=0.0
        for kk in xrange(sum_no_of_batches):
            x1_batch, x2_batch, y_batch, video_lengths = batches.next()
            if len(y_batch)<1:
                continue
            summary, train_batch_correct, train_batch_loss =train_step(x1_batch, x2_batch, y_batch, video_lengths)
            train_writer.add_summary(summary, current_step)
            sum_train_correct = sum_train_correct + train_batch_correct
            train_epoch_loss = train_epoch_loss + train_batch_loss* len(y_batch)
            train_batch_loss_arr.append(train_batch_loss*len(y_batch))
        print("train_loss ={}".format(train_epoch_loss/len(train_set[2])))
        print("total_train_correct={}/total_train={}".format(sum_train_correct, len(train_set[2])))
        train_accuracy.append(sum_train_correct*1.0/len(train_set[2]))
        train_loss.append(train_epoch_loss/len(train_set[2]))


        # Evaluate on Validataion Data for every epoch
        sum_val_correct=0.0
        sum_pos_correct=0.0
        sum_neg_correct=0.0
        sum_pos_samples=0.0
        sum_neg_samples=0.0
        val_epoch_loss=0.0
        val_results = []
        print("\nEvaluation:")
        dev_batches = inpH.batch_iter(dev_set[0],dev_set[1],dev_set[2],dev_set[3], FLAGS.batch_size, 1, convModel_weather.spec, shuffle=False , is_train=False)
        dev_iter=0
        for (x1_dev_b,x2_dev_b,y_dev_b, dev_video_lengths) in dev_batches:
            if len(y_dev_b)<1:
                continue
            dev_iter += 1
            summary, batch_val_correct , val_batch_loss, batch_results = dev_step(x1_dev_b, x2_dev_b, y_dev_b, dev_video_lengths, dev_iter,nn)

            pos_samples = np.sum(y_dev_b)
            sum_pos_samples = sum_pos_samples + pos_samples
            sum_neg_samples= sum_neg_samples + len(y_dev_b)-pos_samples
            pos_correct_array = np.multiply(y_dev_b,batch_results)
            pos_correct=np.sum(pos_correct_array)
            neg_correct=batch_val_correct-pos_correct

            sum_pos_correct = sum_pos_correct + pos_correct
            sum_neg_correct = sum_neg_correct + neg_correct

            val_results = np.concatenate([val_results, batch_results])
            sum_val_correct = sum_val_correct + batch_val_correct


            val_writer.add_summary(summary, current_step)
            val_epoch_loss = val_epoch_loss + val_batch_loss*len(y_dev_b)
            val_batch_loss_arr.append(val_batch_loss*len(y_dev_b))
        print("val_loss ={}".format(val_epoch_loss/len(dev_set[2])))
        print("total_val_correct={}/total_val={}".format(sum_val_correct, len(dev_set[2])))
        print("total_pos_correct={}/total_pos={}".format(sum_pos_correct,sum_pos_samples))
        print("total_neg_correct={}/total_neg={}".format(sum_neg_correct,sum_neg_samples))

        val_accuracy.append(sum_val_correct*1.0/len(dev_set[2]))
        val_loss.append(val_epoch_loss/len(dev_set[2]))
        pos_val_accuracy.append(sum_pos_correct*1.0/sum_pos_samples)
        neg_val_accuracy.append(sum_neg_correct*1.0/sum_neg_samples)



        # Update stored model
        if current_step % (FLAGS.checkpoint_every) == 0:
            max_validation_correct = sum_val_correct
            saver.save(sess, checkpoint_prefix, global_step=current_step)
                #lstm_saver.save(sess, lstm_checkpoint_prefix, global_step=current_step)
            tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
            print("Saved model {} with checkpoint to {}".format(nn, checkpoint_prefix))

        epoch_end_time = time.time()
        empty=[]
        print("Total time for {} th-epoch is {}\n".format(nn, epoch_end_time-epoch_start_time))
        save_plot(train_accuracy, val_accuracy, pos_val_accuracy, neg_val_accuracy, 'epochs', 'accuracy', 'Accuracy vs epochs', [-0.1, nn+0.1, 0, 1.01],  ['train','val','pos_val','neg_val' ],'./accuracy_'+str(FLAGS.name))
        save_plot(train_loss, val_loss,empty,empty, 'epochs', 'loss', 'Loss vs epochs', [-0.1, nn+0.1, 0, np.max(train_loss)+0.2],  ['train','val' ],'./loss_'+str(FLAGS.name))
        save_plot(train_batch_loss_arr, val_batch_loss_arr,empty,empty, 'steps', 'loss', 'Loss vs steps', [-0.1, (nn+1)*sum_no_of_batches+0.1, 0, np.max(train_batch_loss_arr)+0.2],  ['train','val' ],'./loss_batch_'+str(FLAGS.name))

    end_time = time.time()
    print("Total time for {} epochs is {}".format(FLAGS.num_epochs, end_time-start_time))

#"""
