from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
model_dir = __file__[:-3] + "-log"

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)
  confidence = tf.layers.dense(inputs=dropout, units=1, activation=tf.nn.sigmoid)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    print(confidence.shape)
    print(labels.shape)
    print(logits.shape)
    prob = confidence*tf.nn.softmax(logits) + (1.-confidence)*tf.one_hot(labels, 10)
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=prob, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": prob, #tf.nn.softmax(logits, name="softmax_tensor"),
      "confidence": confidence
    }
    
    logits = tf.math.log(tf.clip_by_value(prob, 1e-6, 1.-1e-6))
    """
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
      "confidence": confidence
    }
    """
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) - \
           tf.math.reduce_mean(tf.math.log(1.e-6 + confidence))
    
    mean_c = tf.math.reduce_mean(confidence)
    
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    train_tensors_log = {'global_step': tf.train.get_global_step(),
                         'loss': loss,
                         'mean_c': mean_c,
                        }
    train_hook_list = [tf.train.LoggingTensorHook(
        tensors=train_tensors_log, every_n_iter=1000)]
    #train_hook_list = None#[tf.train.LoggingTensorHook(
    # tensors=, every_n_iter=100)]
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
            training_hooks=train_hook_list)


  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
      "confidence": confidence
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
      "mean_c": tf.metrics.mean(values=confidence)}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True

  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=model_dir,
      config=tf.estimator.RunConfig(#log_step_count_steps=1000,
                                    session_config=session_config))

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  #tensors_to_log = {"probabilities": "softmax_tensor"}
  #eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
  #    x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
  #tensors_to_log = mnist_classifier.evaluate(input_fn=eval_input_fn)
  #logging_hook = tf.train.LoggingMetricHook(
  #    tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  Nfake = 100000
  train_data = np.r_[train_data, np.random.uniform(size=(Nfake, eval_data.shape[1])).astype(eval_data.dtype)]
  train_labels = np.r_[train_labels,
                       (np.random.randint(low=0, high=10,
                        size=Nfake)).astype(train_labels.dtype)]
  train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=128,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=200000)
      #hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
  eval_results1 = mnist_classifier.evaluate(input_fn=eval_input_fn)
  #print(eval_results)
    
  # Evaluate the model using noise and print results
  print(eval_data.shape)
  eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={"x": np.random.uniform(size=(eval_data.shape)).astype(eval_data.dtype)}, y=eval_labels, num_epochs=1, shuffle=False)
  eval_results2 = mnist_classifier.evaluate(input_fn=eval_input_fn)
  #print(eval_results)
      
  # Evaluate the model using noisy images and print results
  print(eval_data.shape)
  eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={"x": np.clip(np.random.uniform(size=(eval_data.shape)).astype(eval_data.dtype) + eval_data, 0., 1.)}, y=eval_labels, num_epochs=1, shuffle=False)
  eval_results3 = mnist_classifier.evaluate(input_fn=eval_input_fn)
  #print(eval_results)

  # Evaluate the model using combined images and print results
  print(eval_data.shape)
  eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={"x": 0.5*(eval_data[:, ::-1] + eval_data)}, y=eval_labels, num_epochs=1, shuffle=False)
  eval_results4 = mnist_classifier.evaluate(input_fn=eval_input_fn)
  #print(eval_results)
  print(eval_results1)
  print(eval_results2)
  print(eval_results3)
  print(eval_results4)


if __name__ == "__main__":
  tf.app.run()

