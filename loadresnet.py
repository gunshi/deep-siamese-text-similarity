import tensorflow as tf
import os
import sys
sys.path.append("/home/tushar/codes/rnn-cnn/models/research/slim/")
sys.path.append("/home/tushar/codes/rnn-cnn/models/research/slim/nets")
slim = tf.contrib.slim
from PIL import Image
from resnet_v2 import *
import numpy as np



checkpoint_file = '/home/tushar/codes/gunsshi/resnet_v2_50.ckpt'
sample_images = ['bus.jpg']



input_tensor = tf.placeholder(tf.float32, shape=(None,None,None,3), name='input_image')
scaled_input_tensor = tf.scalar_mul((1.0/255), input_tensor)
scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)
#Load the model
sess = tf.Session()
arg_scope = resnet_arg_scope()
with slim.arg_scope(arg_scope):
      logits, end_points = resnet_v2_50(scaled_input_tensor,num_classes=0, is_training=False)
      print(end_points.keys())
      saver = tf.train.Saver()
      saver.restore(sess, checkpoint_file)
      for image in sample_images:
            im = Image.open(image)
            width,height=im.size
            im = np.array(im)
            im = im.reshape(-1,height,width,3)
            predict_values, logit_values, pooling2 = sess.run([end_points['Predictions'], logits, end_points['global_pool']], feed_dict={input_tensor: im})
            #print(pooling1)
            #print(pooling1.shape)
            print(pooling2.shape)
            print (np.max(predict_values), np.max(logit_values))
            print (np.argmax(predict_values), np.argmax(logit_values))
