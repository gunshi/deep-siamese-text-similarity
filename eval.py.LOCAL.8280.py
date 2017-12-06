#! /usr/bin/env python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import time
import datetime
#from tensorflow.contrib import learn
from eval_helper_new import InputHelper, compute_distance,plot_precision_recall
from scipy import misc
# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 4)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("model", "/home/tushar/codes/rnn-cnn/runs/testbiggerwindow-450-fulldata/checkpoints/model-6712", "Load trained model checkpoint (Default: None)")
tf.flags.DEFINE_string("eval_filepath", "/home/tushar/Heavy_dataset/mapillary/", "testing folder (default: /home/halwai/gta/final)")
tf.flags.DEFINE_string("ann_filepath", "./annotation_files4/", "testing folde")
tf.flags.DEFINE_integer("max_frames", 20, "Maximum Number of frame (default: 20)")
tf.flags.DEFINE_string("loss", "contrastive", "Type of Loss functions:: contrastive/AAAI(default: contrastive)")
tf.flags.DEFINE_string("name","result" ,"name for saving" )
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("get_stats", False, "get stats")
#tf.flags.DEFINE_string("ann_filepath", "./annotation_files/", "testing folde")
tf.flags.DEFINE_string("pos_file","./newfiles/test-full-pos.txt", "testing folde")
tf.flags.DEFINE_string("neg_file", "./newfiles/test-full-negs.txt", "testing folde")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_filepath==None or FLAGS.model==None :
    print("Eval or Vocab filepaths are empty.")
    exit()

# load data and map id-transform based on training time vocabulary
inpH = InputHelper()


x1_test,x2_test,y_test,video_lengths_test,pairData,reldata = inpH.getTestDataSet(FLAGS.pos_file,FLAGS.neg_file,FLAGS.ann_filepath, FLAGS.eval_filepath, FLAGS.max_frames)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.model
print(checkpoint_file)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)
        # Get the placeholders from the graph by name
        input_imgs = graph.get_operation_by_name("input_imgs").outputs[0]
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        video_lengths = graph.get_operation_by_name("video_lengths").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        conv_output = graph.get_operation_by_name("conv/output").outputs[0]
        predictions = graph.get_tensor_by_name("distance:0")

        print(conv_output, predictions)
        # Generate batches for one epoch
        batches = inpH.batch_iter(x1_test,x2_test,y_test,video_lengths_test,pairData, 1, 1, [[104, 114, 124], (227, 330)] ,shuffle=False, is_train=False) ##??
        # Collect the predictions here
        all_predictions = []
        all_dist=[]


        true_pos=[]
        true_neg=[]
        false_pos=[]
        false_neg=[]

        all_labels=[]
        sum_neg_correct=0.0
        sum_pos_correct=0.0
        batchnum=0
        for (x1_dev_b,x2_dev_b,y_dev_b,v_len_b,pair_data) in batches: #change
            batchnum+=1
            #print(x1_dev_b)
            [x1] = sess.run([conv_output], {input_imgs: x1_dev_b})
            [x2] = sess.run([conv_output], {input_imgs: x2_dev_b})
            [dist] = sess.run([predictions], {input_x1: x1, input_x2: x2, input_y:y_dev_b, dropout_keep_prob: 1.0, video_lengths: v_len_b})
            #diststr=' '.join(dist)
            #ystr=' '.join(y_dev_b)
            misc.imsave('temp/temp'+str(batchnum)+'.png', np.vstack([np.hstack(x1_dev_b),np.hstack(x2_dev_b)]))
            d = compute_distance(dist, FLAGS.loss)
            correct = np.sum(y_dev_b==d)
            correctarr=(y_dev_b==d)
            print(dist, y_dev_b, d)

            for j in range(len(y_dev_b)):
                if(correctarr[j]):
                    if(y_dev_b[j]):
                        true_pos.append(pair_data[j])
                    else:
                        true_neg.append(pair_data[j])
                else:
                    if(y_dev_b[j]):
                        false_neg.append(pair_data[j])
                    else:
                        false_pos.append(pair_data[j])


            num_pos_correct=np.sum(d*correct)
            num_neg_correct=np.sum(correct)-num_pos_correct
            sum_pos_correct=sum_pos_correct+num_pos_correct
            sum_neg_correct=sum_neg_correct+num_neg_correct

            all_dist.append(dist)
            all_predictions.append(correct)
            all_labels.append(y_dev_b)
        #for ex in all_predictions:
        #    print(ex)
        correct_predictions = np.sum(all_predictions)*1.0/ len(all_predictions)
        print("Accuracy: {:g} ".format(correct_predictions))
        total_pos=np.sum(all_labels)
        total_neg=(len(all_labels)-np.sum(all_labels))
        positive_accuracy=sum_pos_correct*1.0/total_pos
        negative_accuracy=sum_neg_correct*1.0/total_neg
        print('total positives:')
        print(total_pos)
        print('positive accuracy')
        print(positive_accuracy)
        print('total negatives:')
        print(total_neg)
        print('negative accuracy')
        print(negative_accuracy)
        #invert dist also

        dist2 = [1-x for x in all_dist]
        precision, recall, _ = precision_recall_curve(all_labels,all_dist)

        precision2,recall2,_=precision_recall_curve(all_labels,dist2)

        plot_precision_recall(recall,precision,'0','./pr_0'+FLAGS.name)
        plot_precision_recall(recall2,precision2,'1','./pr_1'+FLAGS.name)

        print('false neg')
        for i in range(len(false_neg)):
            print(str(false_neg[i][0]))
            print(str(false_pos[i][1]))
            print('')
        print('false pos')
        for i in range(len(false_pos)):
            print(str(false_pos[i][0]))
            print(str(false_pos[i][1]))
            print('')

