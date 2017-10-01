#! /usr/bin/env python3

# Copyright 2017 Daan Wynen
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

import numpy as np
import vgg
from scipy.misc import imread
from scipy.stats import describe
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
slim = tf.contrib.slim

CHECKPOINT_PATH = 'vgg19_normalized'

CAFFE_FEATURES_FNAME = 'features_caffe.npz'
CAFFE_FILTERS_FNAME = 'filters_caffe.npz'
IMAGE_CAT = 'cat.jpg'

# set to true to dump the two .npz files for further inspection
SAVE_OUTPUTS = False
TF_FEATURES_FNAME = 'features_tf.npz'
CAFFE_FEATURES_TF_FNAME = 'features_caffe_tforder.npz'

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

npz = np.load(CAFFE_FEATURES_FNAME)

image_rgb = tf.constant(imread(IMAGE_CAT)[np.newaxis], dtype=tf.float32)
image_rgb_centered = tf.subtract(image_rgb, [_R_MEAN, _G_MEAN, _B_MEAN], 'image_rgb_centered')


# build the tf graph we want to load the weigts and biases into
with slim.arg_scope(vgg.vgg_arg_scope()):
  _, end_points = vgg.vgg_19(image_rgb_centered,
                             is_training=False,
                             spatial_squeeze=False)
  

# get the variables for the convolutions.
# the caffemodel does not contain the fully-connected layers,
# so we ignore these here
vgg_conv_variables = {v.name: v for v in slim.get_model_variables('vgg_19') if '/fc' not in v.name}

print('reading caffe filters')
caffe_filters = np.load(CAFFE_FILTERS_FNAME)
assignments = dict()
for k, v in sorted(caffe_filters.items(), key=lambda p: p[0]):
  block = k[:5]
  target_var = vgg_conv_variables['vgg_19/{}/{}:0'.format(block, k)]
  if target_var.name.endswith('biases:0'):
    assignments[target_var.name] = v
  else:
    assert target_var.name.endswith('weights:0')
    assignments[target_var.name] = v.transpose(2, 3, 1, 0)

# invert the pixel order for the first convolution to go from BGR to RGB inputs
assignments['vgg_19/conv1/conv1_1/weights:0'] = assignments['vgg_19/conv1/conv1_1/weights:0'][:, :, ::-1, :]

assign_op, assign_feed = slim.assign_from_values(assignments)

print('reading caffe activations')
caffe_npz = np.load(CAFFE_FEATURES_FNAME)

activation_targets = dict()
caffe_activations = dict()
for k, v in sorted(caffe_npz.items(), key=lambda p: p[0]):
  if k.startswith('pool'):
    t_name = 'vgg_19/'+k
    caffe_activations[t_name] = v.transpose(0, 2, 3, 1)
    activation_targets[t_name] = end_points[t_name]
  elif k == 'data':
    continue
  else:
    block = k[:5]
    ep_name = 'vgg_19/{}/{}'.format(block, k)
    caffe_activations[ep_name] = v.transpose(0, 2, 3, 1)
    activation_targets[ep_name] = end_points[ep_name]


with tf.Session() as sess:
  print('initializing')
  sess.run(tf.global_variables_initializer())

  print('assigning caffe weights')
  sess.run(assign_op, feed_dict=assign_feed)

  print('saving checkpoint to {}'.format(CHECKPOINT_PATH))
  # we only save the variables we're after
  saver = tf.train.Saver(list(vgg_conv_variables.values()))
  saver.save(sess, save_path=CHECKPOINT_PATH)

  tf_activations = sess.run(activation_targets)

if SAVE_OUTPUTS:
  np.savez(TF_FEATURES_FNAME, **activations)
  np.savez(CAFFE_FEATURES_TF_FNAME, **caffe_activations)

# print some stats on how well the activations at different layers match
for k in sorted(caffe_activations.keys()):
  print(k)
  ck = caffe_activations[k]
  tk = tf_activations[k]
  diff = np.abs((ck - tk))
  print(describe(diff.flatten()))
  print(np.percentile(diff, 99))
  print()
  
print('DONE')
