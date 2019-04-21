import tensorflow as tf
import numpy as np


class SDC_CNN:
    def __init__(self, input_dim, epoch=250, learning_rate=0.001):

        """        
        :param input_dim: dimension of one input
        :param epoch: number of time we run through the data
        :param learning_rate: learning rate for the gradient descent"""
        
        # Hyperparameters
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.x = tf.placeholder(dtype=tf.float32, shape=[None]+self.input_dim)
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # Weights for the layers
        W1 = tf.Variable(tf.random_normal([5, 5, 3, 24]))
        b1 = tf.Variable(tf.random_normal([24]))

        W2 = tf.Variable(tf.random_normal([5, 5, 24, 36]))
        b2 = tf.Variable(tf.random_normal([36]))

        W3 = tf.Variable(tf.random_normal([5, 5, 36, 48]))
        b3 = tf.Variable(tf.random_normal([48]))

        W4 = tf.Variable(tf.random_normal([3, 3, 48, 64]))
        b4 = tf.Variable(tf.random_normal([64]))

        W5 = tf.Variable(tf.random_normal([3, 3, 64, 64]))
        b5 = tf.Variable(tf.random_normal([64]))

        W6 = tf.Variable(tf.random_normal([6144, 100]))
        b6 = tf.Variable(tf.random_normal([100]))

        W7 = tf.Variable(tf.random_normal([100, 50]))
        b7 = tf.Variable(tf.random_normal([50]))

        W8 = tf.Variable(tf.random_normal([50, 10]))
        b8 = tf.Variable(tf.random_normal([10]))

        Wout = tf.Variable(tf.random_normal([10, 1]))
        bout = tf.Variable(tf.random_normal([1]))

        def conv_layer(x, W, b, strides):
            conv = tf.nn.conv2d(x, W, strides=strides, padding='VALID')
            conv_with_b = tf.nn.bias_add(conv, b)
            conv_out = tf.nn.relu(conv_with_b)
            return conv_out


    	# Normalization layer
        self.x_normalized = tf.subtract(tf.divide(self.x, 127.5), 1)

        # Convolution layers
        layer_1 = conv_layer(self.x_normalized, W1, b1, [1, 2, 2, 1])
        layer_2 = conv_layer(layer_1, W2, b2, [1, 2, 2, 1])
        layer_3 = conv_layer(layer_2, W3, b3, [1, 2, 2, 1])
        layer_4 = conv_layer(layer_3, W4, b4, [1, 2, 2, 1])
        layer_5 = conv_layer(layer_4, W5, b5, [1, 1, 1, 1])

    	# Fully connected layers
        flatten_layer = tf.reshape(layer_5, [-1, W6.get_shape().as_list()[0]])
        layer_6 = tf.nn.relu(tf.add(tf.matmul(flatten_layer, W6), b6))
        layer_7 = tf.nn.relu(tf.add(tf.matmul(layer_6, W7), b7))
        layer_8 = tf.nn.relu(tf.add(tf.matmul(layer_7, W8), b8))
        self.output_value = tf.nn.relu(tf.add(tf.matmul(layer_8, Wout), bout))

    	# Loss function and optimizer
        self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.output_value)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

		# Object to save the model
        self.saver = tf.train.Saver()

    def train(self, data, true_values):

        """Train the model with sample data

           :param data: a set of training data -> 4D array
           :param true_values: the true values for the supervised learning"""

        num_samples = len(data[:,1,1,1])

        # Open tensorflow session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Run through epochs
            for i in range(self.epoch):
            	# Run through samples
                for j in range(num_samples):
                    sample = data[j,:,:,:].reshape(1, self.input_dim[0], self.input_dim[1], self.input_dim[2])
                    sample_true_value = true_values[j].reshape(1,1)
                    o, l, _ = sess.run([self.output_value, self.loss, self.train_op],feed_dict={self.x: sample, self.y: sample_true_value})
                print('epoch {0}: loss = {1}'.format(i, l))
            self.saver.save(sess, './model.ckpt')

    def predict(self, data):

        """Predict the value for a new data

        :param data: a new data -> 3D array"""

        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')
            output = sess.run(self.output_value, feed_dict={self.x: data})
        print('output', output)
        return output



