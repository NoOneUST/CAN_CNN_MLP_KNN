from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math
batch_size=64
max_epoch=20
channel_set=[1,2,4,8,16,32,64,128,256,512,1024]

def deepnn(x,channel):
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([3, 3, 1, channel])
    b_conv1 = bias_variable([channel])
    h_conv1 = tf.nn.leaky_relu(tf.nn.atrous_conv2d(x_image, filters=W_conv1, rate=1, padding='SAME')+b_conv1)
    #h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, channel, channel])
    b_conv2 = bias_variable([channel])
    h_conv2 = tf.nn.leaky_relu(tf.nn.atrous_conv2d(h_conv1, filters=W_conv2, rate=2, padding='SAME')+b_conv2)
    #h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  with tf.name_scope('conv3'):
    W_conv3 = weight_variable([3, 3, channel, channel])
    b_conv3 = bias_variable([channel])
    h_conv3 = tf.nn.leaky_relu(tf.nn.atrous_conv2d(h_conv2, filters=W_conv3, rate=4, padding='SAME')+b_conv3)
    #h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  with tf.name_scope('conv4'):
    W_conv4 = weight_variable([3, 3, channel, channel])
    b_conv4 = bias_variable([channel])
    h_conv4 = tf.nn.leaky_relu(tf.nn.atrous_conv2d(h_conv3, filters=W_conv4, rate=8, padding='SAME')+b_conv4)
    #h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  with tf.name_scope('conv5'):
    W_conv5 = weight_variable([3, 3, channel, 10])
    b_conv5 = bias_variable([10])
    h_conv5 = tf.nn.leaky_relu(tf.nn.atrous_conv2d(h_conv4, filters=W_conv5, rate=1, padding='SAME')+b_conv5)
    #h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  with tf.name_scope('global_aver_pool'):
    y_conv = tf.keras.layers.GlobalAveragePooling2D()(h_conv5)

  return y_conv

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main():
  # Import data
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])
  accuset=[]

  for channel in channel_set:
    # Build the graph for the deep net
    y_conv = deepnn(x,channel)

    with tf.name_scope('loss'):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                              logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
      train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    train_writer = tf.summary.FileWriter("model")
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(math.floor(mnist.train.num_examples*max_epoch/batch_size)):
        batch = mnist.train.next_batch(batch_size)
        if i % 100 == 0:
          train_accuracy = accuracy.eval(feed_dict={
              x: batch[0], y_: batch[1]})
          print(('step %d, training accuracy %g' % (i, train_accuracy)))
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
      accu_value=accuracy.eval(feed_dict={
          x: mnist.test.images, y_: mnist.test.labels})
      print(('Channel %d, test accuracy %g' % (channel,accu_value)))

      accuset.append(accu_value)
  print(channel_set)
  print(accu_value)
if __name__ == '__main__':
    main()