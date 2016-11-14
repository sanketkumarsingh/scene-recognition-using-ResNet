# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Builds the CIFAR-10 network.
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input
from tensorflow.python import control_flow_ops
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 10 ,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './batch',
                           """Path to the SUN 397 data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.b
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01    # Initial learning rate.

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'train')
  print ('data_dir:',data_dir)
  return cifar10_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'test')
  return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=FLAGS.batch_size)



# basenet
def inference2(images):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  keep_prob = tf.Variable(0.5)
  h_fc1_drop = tf.nn.dropout(pool2, keep_prob)

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    dim = 1
    for d in h_fc1_drop.get_shape()[1:].as_list():
      dim *= d
    reshape = tf.reshape(h_fc1_drop, [FLAGS.batch_size, dim])

    weights = _variable_with_weight_decay('weights', shape=[dim, 1024],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)


  h_fc2_drop = tf.nn.dropout(local3, keep_prob)
  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[1024, 512],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(h_fc2_drop, weights) + biases, name=scope.name)
    _activation_summary(local4)

  h_fc3_drop = tf.nn.dropout(local4, keep_prob)
  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [512, NUM_CLASSES],
                                          stddev=1 / 512.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    #softmax_linear = tf.nn.softmax(tf.matmul(local4, weights) + biases, name=scope.name)
    softmax_linear = tf.add(tf.matmul(h_fc3_drop, weights), biases, name=scope.name)

    _activation_summary(softmax_linear)

  return softmax_linear

#alexnet
def inference3(images):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5,5, 3, 48],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [48], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # norm1
  norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
  # pool1
  pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')


  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 48, 128],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 256],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  #conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv4)

  #conv5
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 128],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv5)

  # pool3
  pool3 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  keep_prob = tf.Variable(0.5)
  h_fc1_drop = tf.nn.dropout(pool3, keep_prob)

  # FC1
  with tf.variable_scope('fc1') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    dim = 1
    for d in h_fc1_drop.get_shape()[1:].as_list():
      dim *= d
    reshape = tf.reshape(h_fc1_drop, [FLAGS.batch_size, dim])

    weights = _variable_with_weight_decay('weights', shape=[dim, 1024],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
    fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(fc1)

  h_fc2_drop = tf.nn.dropout(fc1, keep_prob)

  # FC2
  with tf.variable_scope('fc2') as scope:
    weights = _variable_with_weight_decay('weights', shape=[1024, 1024],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
    fc2 = tf.nn.relu(tf.matmul(h_fc2_drop, weights) + biases, name=scope.name)
    _activation_summary(fc2)

  h_fc3_drop = tf.nn.dropout(fc2, keep_prob)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [1024, NUM_CLASSES],
                                          stddev=0.04 , wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(h_fc3_drop, weights), biases, name=scope.name)
    #softmax_linear = tf.nn.softmax(tf.matmul(fc2, weights) + biases, name=scope.name)

    _activation_summary(softmax_linear)

  return softmax_linear

#resnet
def inference(images):
    """Build the CIFAR-10 model.
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    num_conv = 2
    layers = []
    with tf.variable_scope('conv1') as scope:
        conv1 = conv_layer(images, [3, 3, 3, 16], 1 , scope,'first')
        layers.append(conv1)

    for i in range(num_conv):
        with tf.variable_scope('conv2_x_%d' % (i + 1)) as scope:
            conv2_x = residual_block(layers[-1], 16, False, scope)
            layers.append(conv2_x)
        with tf.variable_scope('conv2_%d' % (i+1) ) as scope:
            conv2 = residual_block(layers[-1], 16, False, scope)
            layers.append(conv2)

        assert conv2.get_shape().as_list()[1:] == [100, 100, 16]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv3_x_%d' % (i + 1)) as scope:
            conv3_x = residual_block(layers[-1], 32, down_sample, scope)
            layers.append(conv3_x)
        with tf.variable_scope('conv3_%d' % (i+1) ) as scope:
            conv3 = residual_block(layers[-1], 32, False, scope)
            layers.append(conv3)

        assert conv3.get_shape().as_list()[1:] == [50, 50, 32]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.variable_scope('conv4_x_%d' % (i + 1)) as scope:
            conv4_x = residual_block(layers[-1], 64, down_sample, scope)
            layers.append(conv4_x)
        with tf.variable_scope('conv4_%d' % (i+1) ) as scope:
            conv4 = residual_block(layers[-1], 64, False, scope)
            layers.append(conv4)

        assert conv4.get_shape().as_list()[1:] == [25, 25, 64]

    keep_prob = tf.constant(0.5,dtype='float32')
    h_fc1_drop = tf.nn.dropout(layers[-1], keep_prob)
    layers.append(h_fc1_drop)

    with tf.variable_scope('fc1') as scope:

        last_layer = layers[-1]
        dim = 1
        for d in last_layer.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(last_layer, [FLAGS.batch_size, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, 1024],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name )
        _activation_summary(fc1)

    h_fc2_drop = tf.nn.dropout(fc1, keep_prob)

    with tf.variable_scope('fc2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1024, NUM_CLASSES],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(h_fc2_drop, weights) + biases, name=scope.name)
        _activation_summary(fc2)

    h_fc3_drop = tf.nn.dropout(fc2, keep_prob)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [NUM_CLASSES, NUM_CLASSES], stddev=0.04, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        soft_max = tf.add(tf.matmul(h_fc3_drop, weights), biases, name=scope.name)
        #soft_max = tf.nn.softmax(tf.matmul(fc2, weights) + biases, name=scope.name)
        _activation_summary(soft_max)
    return soft_max, layers[8], layers[12] , weights



def conv_layer(inpt, filter_shape, stride , scope, first):
    depth = filter_shape[3]
    kernel = _variable_with_weight_decay(scope.name+first+'weights', shape=filter_shape,
                                         stddev=1e-4, wd=0.0)

    conv = tf.nn.conv2d(inpt, kernel, [stride, stride, stride, stride], padding='SAME')
    biases = _variable_on_cpu(scope.name+first+'biases', [depth], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)

    # batch normalization
    beta = tf.constant(0.0, shape=[depth])
    gamma = tf.constant(1.0, shape=[depth])

    batch_mean, batch_var = tf.nn.moments(bias, [0, 1, 2], name='moments')

    conv_bn = tf.nn.batch_norm_with_global_normalization(bias, batch_mean, batch_var, beta, gamma, 1e-3, True)

    conv1 = tf.nn.relu(conv_bn)
    return conv1


def residual_block(inpt, output_depth, down_sample, scope,projection=False):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1,2,2,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')
    first = 'first'
    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], 1,scope , first)
    first = 'second'
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1 , scope,first)

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt



    res = conv2 + input_layer
    return res


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  dest_directory = FLAGS.data_dir
  # if not os.path.exists(dest_directory):
  #   os.makedirs(dest_directory)
  # filename = DATA_URL.split('/')[-1]
  # filepath = os.path.join(dest_directory, filename)
  # if not os.path.exists(filepath):
  #   def _progress(count, block_size, total_size):
  #     sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
  #         float(count * block_size) / float(total_size) * 100.0))
  #     sys.stdout.flush()
  #   filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
  #                                            reporthook=_progress)
  #   print()
  #   statinfo = os.stat(filepath)
  #   print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  #   tarfile.open(filepath, 'r:gz').extractall(dest_directory)
