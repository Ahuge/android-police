import tensorflow
import numpy as np
import datetime

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")


def generator(batch_size, z_dim):
    z = tensorflow.random_normal(
        [batch_size, z_dim], mean=0, stddev=1, name='z'
    )

    g_w1 = tensorflow.get_variable(
        'g_w1',
        [z_dim, 3136],
        dtype=tensorflow.float32,
        initializer=tensorflow.truncated_normal_initializer(stddev=0.02)
    )
    g_b1 = tensorflow.get_variable(
        'g_b1',
        [3136],
        initializer=tensorflow.truncated_normal_initializer(stddev=0.02)
    )

    g1 = tensorflow.matmul(z, g_w1) + g_b1
    g1 = tensorflow.reshape(g1, [-1, 56, 56, 1])
    g1 = tensorflow.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tensorflow.nn.relu(g1)

    # Generate 50 features
    g_w2 = tensorflow.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tensorflow.float32, initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
    g_b2 = tensorflow.get_variable('g_b2', [z_dim/2], initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
    g2 = tensorflow.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tensorflow.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tensorflow.nn.relu(g2)
    g2 = tensorflow.image.resize_images(g2, [56, 56])

    # Generate 25 features
    g_w3 = tensorflow.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tensorflow.float32, initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
    g_b3 = tensorflow.get_variable('g_b3', [z_dim/4], initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
    g3 = tensorflow.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tensorflow.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tensorflow.nn.relu(g3)
    g3 = tensorflow.image.resize_images(g3, [56, 56])

    # Final convolution with one output channel
    g_w4 = tensorflow.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tensorflow.float32, initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
    g_b4 = tensorflow.get_variable('g_b4', [1], initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
    g4 = tensorflow.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tensorflow.sigmoid(g4)

    # No batch normalization at the final layer, but we do add
    # a sigmoid activator to make the generated images crisper.
    # Dimensions of g4: batch_size x 28 x 28 x 1

    return g4