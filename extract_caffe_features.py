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

import sys
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import hashlib
import numpy as np
from scipy.misc import imread
import json

PROTOTXT_FNAME = 'VGG_ILSVRC_19_layers_deploy_fullconv.prototxt'
CAFFEMODEL_FNAME = 'vgg_normalised.caffemodel'


CAFFE_FEATURES_FNAME = 'features_caffe.npz'
CAFFE_FILTERS_FNAME = 'filters_caffe.npz'
IMAGE_CAT = 'cat.jpg'

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

expected_hash_prototxt = 'dc965efaff95dc64360c84ad4bce13d03f6cc062f9e8d2e40428e0c407360fe55f6a91d54980ea70ac6fdd6ccd724685e2a74fb145bf7e8fc0c954497a85357b'
expected_hash_caffemodel = '63f1af8188923a09fc0fd9c111ae7085e0a7045e9ddf2b6f97164e995034b65dbcebac4360bde154e6e771b16c4d6a32a4986fd809696d6a9d10f530271f6e15'

print('checking prototxt hash')
with open(PROTOTXT_FNAME, 'rb') as f:
  hash = hashlib.sha512(b''.join(f.readlines())).hexdigest()
  assert hash == expected_hash_prototxt
print('checking caffemodel hash')
with open(CAFFEMODEL_FNAME, 'rb') as f:
  hash = hashlib.sha512(b''.join(f.readlines())).hexdigest()
  assert hash == expected_hash_caffemodel
print('ready to go')

image = imread(IMAGE_CAT)
net = caffe.Net(PROTOTXT_FNAME, CAFFEMODEL_FNAME,
                caffe.TEST)

# remove mean channels, move channels to outermost dimension, and insert dimension for batch size 1
transformed_image = (image - [_R_MEAN, _G_MEAN, _B_MEAN]).transpose(2, 0, 1)[np.newaxis, ::-1, :, :]

net.blobs['data'].data[...] = transformed_image
blob_names = list(net.blobs.keys())
results = net.forward_all(blob_names, data=transformed_image)


np.savez(CAFFE_FEATURES_FNAME, **results)
weights = dict()
for k, v in net.params.items():
  print('{}\tbiases: {}\tweights:{}'.format(k, v[1].data.shape, v[0].data.shape))
  weights[k+'/weights'] = v[0].data
  weights[k+'/biases'] = v[1].data
  assert weights[k+'/weights'].ndim == 4
  assert weights[k+'/biases'].ndim == 1

np.savez(CAFFE_FILTERS_FNAME, **weights)
