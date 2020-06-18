# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
from model import create_model
from solver import create_admm_solver
from tensorflow.examples.tutorials.mnist import input_data
from prune_utility import apply_prune_on_grads,apply_prune,get_configuration,projection
import tensorflow as tf
import numpy as np
from numpy import linalg as LA
import tensorflow as tf
#import tensorflow_datasets as tfds
import numpy as np
import random
import time

#tf.disable_eager_execution()

FLAGS = None
# pruning ratio

data_name_list = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']

data_name = data_name_list[3]
num_class = 100

prune_configuration = get_configuration()
dense_w = {}
P1 = prune_configuration.P1
P2 = prune_configuration.P2
P3 = prune_configuration.P3
P4 = prune_configuration.P4

prune_configuration.display()


def main(_):
  # Import data
  train_image = np.load('C:\\Users\\Yang\\Desktop\\admm-pruning-master\\data\\'+data_name+'_train_image'+'.npy')
  train_label = np.load('C:\\Users\\Yang\\Desktop\\admm-pruning-master\\data\\'+data_name+'_train_label'+'.npy')
  test_image = np.load('C:\\Users\\Yang\\Desktop\\admm-pruning-master\\data\\'+data_name+'_test_image'+'.npy')
  test_label = np.load('C:\\Users\\Yang\\Desktop\\admm-pruning-master\\data\\'+data_name+'_test_label'+'.npy')

  model = create_model()
  x = model.x
  y_ = model.y_
  cross_entropy = model.cross_entropy
  layers = model.layers
  logits = model.logits
  solver = create_admm_solver(model)
  keep_prob = model.keep_prob
  train_step = solver.train_step
  train_step1 = solver.train_step1
  
  W_conv1 = model.W_conv1
  W_conv2 = model.W_conv2
  W_fc1 = model.W_fc1
  W_fc2 = model.W_fc2
  
  A = solver.A
  B = solver.B
  C = solver.C
  D = solver.D
  E = solver.E
  F = solver.F
  G = solver.G
  H = solver.H
    

  # 첫 번째 stop condition
  stop_W_Z_Conv1 = [] 
  stop_W_Z_Conv2 = [] 
  stop_W_Z_FC1 = []
  stop_W_Z_FC2 = []
  stop_W_Z = [] #전체 값  
    
  # 두 번째 stop condition
  stop_Z_Z_Conv1 = [] 
  stop_Z_Z_Conv2 = [] 
  stop_Z_Z_FC1 = [] 
  stop_Z_Z_FC2 = [] 
  stop_W_Z = [] #전체 값  
    
  time_list = []
  
    
  my_trainer = tf.train.AdamOptimizer(1e-3)
  grads = my_trainer.compute_gradients(cross_entropy)
    
  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        j = random.randrange(0, len(train_image), 1)
        batch = []
        batch.append(train_image[j])
        batch.append(train_label[j])
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    start_pre = time.time()
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: test_image, y_: test_label, keep_prob: 1.0}))
    time_pre = time.time() - start_pre
    time_list.append(time_pre)

    pretrained_weight = np.array([sess.run(W_conv1), sess.run(W_conv2), sess.run(W_fc1), sess.run(W_fc2)])
    
    # pretrain된 weight 저장 
    np.save('C:\\Users\\Yang\\Desktop\\admm-pruning-master\\weight\\'+'pretrained_'+data_name, pretrained_weight)
    
    Z1 = sess.run(W_conv1)
    Z1 = projection(Z1, percent=P1)

    U1 = np.zeros_like(Z1)

    Z2 = sess.run(W_conv2)
    Z2 = projection(Z2, percent=P2)

    U2 = np.zeros_like(Z2)

    Z3 = sess.run(W_fc1)
    Z3 = projection(Z3, percent=P3)

    U3 = np.zeros_like(Z3)

    Z4 = sess.run(W_fc2)
    Z4 = projection(Z4, percent=P4)

    U4 = np.zeros_like(Z4)
    
    for j in range(50):
        for i in range(5000):
            kk = random.randrange(0, len(train_image), 1)
            batch = []
            batch.append(train_image[kk])
            batch.append(train_label[kk])
            
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step1.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0, A:Z1, B:U1, C:Z2, D:U2, E:Z3, F:U3, G:Z4, H:U4})
        
        #Z k번째값 저장
        Z_k = [Z1, Z2, Z3, Z4]
        
        Z1 = sess.run(W_conv1) + U1
        Z1 = projection(Z1, percent=P1)

        U1 = U1 + sess.run(W_conv1) - Z1

        Z2 = sess.run(W_conv2) + U2
        Z2 = projection(Z2, percent=P2)

        U2 = U2 + sess.run(W_conv2) - Z2

        Z3 = sess.run(W_fc1) + U3
        Z3 = projection(Z3, percent=P3)

        U3 = U3 + sess.run(W_fc1) - Z3

        Z4 = sess.run(W_fc2) + U4
        Z4 = projection(Z4, percent=P4)

        U4 = U4 + sess.run(W_fc2) - Z4
        
        
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: test_image, y_: test_label, keep_prob: 1.0}))
        print('============= [' + str(j)+'] gogo ==========')
        #stop condition 값 저장 
        stop_W_Z_Conv1.append(LA.norm(sess.run(W_conv1) - Z1))
        stop_W_Z_Conv2.append(LA.norm(sess.run(W_conv2) - Z2))
        stop_W_Z_FC1.append(LA.norm(sess.run(W_fc1) - Z3))
        stop_W_Z_FC2.append(LA.norm(sess.run(W_fc2) - Z4))
        
        
        stop_Z_Z_Conv1.append(LA.norm(Z_k[0]-Z1))
        stop_Z_Z_Conv2.append(LA.norm(Z_k[1]-Z2))
        stop_Z_Z_FC1.append(LA.norm(Z_k[2]-Z3))
        stop_Z_Z_FC2.append(LA.norm(Z_k[3]-Z4))
        
#         print(LA.norm(sess.run(W_conv1) - Z1))
#         print(LA.norm(sess.run(W_conv2) - Z2))
#         print(LA.norm(sess.run(W_fc1) - Z3))
#         print(LA.norm(sess.run(W_fc2) - Z4))

    
    
    stop_W_Z = [stop_W_Z_Conv1, stop_W_Z_Conv2, stop_W_Z_FC1, stop_W_Z_FC2]
    stop_Z_Z = [stop_Z_Z_Conv1, stop_Z_Z_Conv2, stop_Z_Z_FC1, stop_Z_Z_FC2]
    
    np.save('C:\\Users\\Yang\\Desktop\\admm-pruning-master\\convergence\\'+data_name+'_stopWZ', stop_W_Z)
    np.save('C:\\Users\\Yang\\Desktop\\admm-pruning-master\\convergence\\'+data_name+'_stopZZ', stop_Z_Z)
    
    
    dense_w['conv1/W_conv1'] = W_conv1
    dense_w['conv2/W_conv2'] = W_conv2
    dense_w['fc1/W_fc1'] = W_fc1
    dense_w['fc2/W_fc2'] = W_fc2
    
    dict_nzidx = apply_prune(dense_w,sess)
    admm_weight = np.array([sess.run(W_conv1), sess.run(W_conv2), sess.run(W_fc1), sess.run(W_fc2)])
    np.save('C:\\Users\\Yang\\Desktop\\admm-pruning-master\\weight\\'+'admm_'+data_name, admm_weight)
    
    
    print ("checking space dictionary")
    print (dict_nzidx.keys())
    grads = apply_prune_on_grads(grads,dict_nzidx)
    apply_gradient_op = my_trainer.apply_gradients(grads)
    for var in tf.global_variables():
                if tf.is_variable_initialized(var).eval() == False:
                    sess.run(tf.variables_initializer([var]))
    print ("start retraining after pruning")
    for i in range(10000):
        j = random.randrange(0, len(train_image), 1)
        batch = []
        batch.append(train_image[j])
        batch.append(train_label[j])
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))

        apply_gradient_op.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    retrain_weight = np.array([sess.run(W_conv1), sess.run(W_conv2), sess.run(W_fc1), sess.run(W_fc2)])
    np.save('C:\\Users\\Yang\\Desktop\\admm-pruning-master\\weight\\'+'retrained_'+data_name, retrain_weight)
    
    start_re = time.time()
    print('test accuracy %g' % accuracy.eval(feed_dict={
          x: test_image, y_: test_label, keep_prob: 1.0}))
    time_re = time.time() - start_re
    time_list.append(time_re)
    
    np.save('C:\\Users\\Yang\\Desktop\\admm-pruning-master\\time\\'+data_name+'_time', time_list)
    
    print(np.sum(sess.run(W_conv1)!=0))
    print(np.sum(sess.run(W_conv2) != 0))
    print(np.sum(sess.run(W_fc1) != 0))
    print(np.sum(sess.run(W_fc2) != 0))
    # do the saving.
    saver = tf.train.Saver()
    saver.save(sess,"./lenet_5_pruned_model.ckpt")
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  
