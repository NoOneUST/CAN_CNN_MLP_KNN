from __future__ import print_function

import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter

# Import MNIST data

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

Xtr, Ytr = mnist.train.next_batch(mnist.train.num_examples)

Xte, Yte = mnist.test.next_batch(mnist.test.num_examples)

# tf Graph Input

xtr = tf.placeholder("float", [None, 784])

xte = tf.placeholder("float", [784])

# pred = tf.arg_min(distance, 0)

accuRes = []

# Start training

for K in range(1, 11):

    with tf.Session() as sess:

        # Nearest Neighbor calculation using L1 Distance

        # Calculate L1 Distance

        distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
        accuracy = 0.

        # Prediction: Get min distance index (Nearest neighbor)
        pred_k = tf.nn.top_k(-distance, K).indices

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Run the initializer

        sess.run(init)

        # loop over test data

        for i in range(len(Xte)):

            # Get nearest neighbor
            preK_y=Ytr[sess.run(pred_k, feed_dict={xtr: Xtr, xte: Xte[i, :]})]
            predict_K=[]
            for j in preK_y:
                predict_K.append(np.argmax(j))
            predNumCounter=Counter(predict_K)
            predict_Num=predNumCounter.most_common(1)[0][0]

            # Get nearest neighbor class label and compare it to its true label

            # print(sess.run(pred_k, feed_dict={xtr: Xtr, xte: Xte[i, :]}))
            # print(preK_y)
            # print(predict_K)
            # print(predict_Num)
            # print(np.argmax(Yte[i]))
            print("Test", i, "Prediction:", predict_Num,
                  "True Class:", np.argmax(Yte[i]))

            # Calculate accuracy

            if predict_Num == np.argmax(Yte[i]):
                accuracy += 1. / len(Xte)

        print("k=", K, "Accuracy:", accuracy)
        accuRes.append(accuracy)
x_axis = range(1, 11)
plt.figure()
plt.plot(x_axis, accuRes)
plt.show()