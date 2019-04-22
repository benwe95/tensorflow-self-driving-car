import tensorflow as tf
import numpy as np
import math


class SDC_CNN:
    def __init__(self, input_dim, epochs=200, learning_rate=0.0001):

        """        
        :param input_dim: dimension of one input
        :param epoch: number of time we run through the data
        :param learning_rate: learning rate for the gradient descent"""
        
        # Hyperparameters
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.L2_norm = 0.001

        self.x = tf.placeholder(dtype=tf.float32, shape=[None]+self.input_dim)
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        def conv_layer(x, W, b, strides):
            conv = tf.nn.conv2d(x, W, strides=strides, padding='VALID')
            conv_with_b = tf.nn.bias_add(conv, b)
            conv_out = tf.nn.relu(conv_with_b)
            return conv_out

        # normalization layer
        self.x_normalized = tf.subtract(tf.divide(self.x, 127.5), 1)

        # convolution layers
        W_conv_1 = tf.Variable(tf.random_normal([5, 5, 3, 24]))
        b_conv_1 = tf.Variable(tf.random_normal([24]))
        conv_layer_1 = conv_layer(self.x_normalized, W_conv_1, b_conv_1, [1, 2, 2, 1])

        W_conv_2 = tf.Variable(tf.random_normal([5, 5, 24, 36]))
        b_conv_2 = tf.Variable(tf.random_normal([36]))      
        conv_layer_2 = conv_layer(conv_layer_1, W_conv_2, b_conv_2, [1, 2, 2, 1])

        W_conv_3 = tf.Variable(tf.random_normal([5, 5, 36, 48]))
        b_conv_3 = tf.Variable(tf.random_normal([48]))
        conv_layer_3 = conv_layer(conv_layer_2, W_conv_3, b_conv_3, [1, 2, 2, 1])

        W_conv_4 = tf.Variable(tf.random_normal([3, 3, 48, 64]))
        b_conv_4 = tf.Variable(tf.random_normal([64]))
        conv_layer_4 = conv_layer(conv_layer_3, W_conv_4, b_conv_4, [1, 1, 1, 1])

        W_conv_5 = tf.Variable(tf.random_normal([3, 3, 64, 64]))
        b_conv_5 = tf.Variable(tf.random_normal([64]))
        conv_layer_5 = conv_layer(conv_layer_4, W_conv_5, b_conv_5, [1, 1, 1, 1])

        # flatten_layer
        flatten_layer = tf.reshape(conv_layer_5, [-1, 1152])

        # fully connected layers
        W_fc_1 = tf.Variable(tf.random_normal([1152, 1164]))
        b_fc_1 = tf.Variable(tf.random_normal([1164]))
        fc_layer_1 = tf.nn.relu(tf.add(tf.matmul(flatten_layer, W_fc_1), b_fc_1))

        W_fc_2 = tf.Variable(tf.random_normal([1164, 100]))
        b_fc_2 = tf.Variable(tf.random_normal([100]))
        fc_layer_2 = tf.nn.relu(tf.add(tf.matmul(fc_layer_1, W_fc_2), b_fc_2))

        W_fc_3 = tf.Variable(tf.random_normal([100, 50]))
        b_fc_3 = tf.Variable(tf.random_normal([50]))
        fc_layer_3 = tf.nn.relu(tf.add(tf.matmul(fc_layer_2, W_fc_3), b_fc_3))

        W_fc_4 = tf.Variable(tf.random_normal([50, 10]))
        b_fc_4 = tf.Variable(tf.random_normal([10]))
        fc_layer_4 = tf.nn.relu(tf.add(tf.matmul(fc_layer_3, W_fc_4), b_fc_4))

        W_fc_5 = tf.Variable(tf.random_normal([10, 1]))
        b_fc_5 = tf.Variable(tf.random_normal([1]))
        self.output_value =  tf.divide(tf.atan(tf.matmul(fc_layer_4, W_fc_5) + b_fc_5), (math.pi/2))

    	# loss function and optimizer
        variables = tf.trainable_variables()
        self.loss = tf.add(tf.reduce_mean(tf.square(tf.subtract(self.y, self.output_value))), 
        tf.add_n([tf.nn.l2_loss(variable) for variable in variables]) * self.L2_norm)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

		# object to save the model
        self.saver = tf.train.Saver()

    def get_batch(self, X, true_values, size):
        a = np.random.choice(len(X), size, replace=False)
        return (X[a], true_values[a].reshape(size,1))

    def train(self, data, true_values, batch_size=400):

        """Train the model with sample data

           :param data: a set of training data -> 4D array
           :param true_values: the true values for the supervised learning
           :param batch size: the size of the batch for one iteration"""

        num_samples = len(data[:,1,1,1])

        # open tensorflow session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # run through epochs
            for epoch in range(self.epochs):
            	# Run through dataset
                for i in range(int(num_samples/batch_size)):
                    batch_data, batch_true_values = self.get_batch(data, true_values, batch_size)
                    o, l, _ = sess.run([self.output_value, self.loss, self.train_op],
                                        feed_dict={self.x: batch_data, self.y: batch_true_values})
                    if i % 10 == 0:
                        print('epoch {0}: loss = {1}'.format(i, l))

            self.saver.save(sess, './model.ckpt')

    def predict(self, data):

        """Predict the value for a new data

        :param data: a new data -> 3D array"""

        with tf.Session() as sess:
            self.saver.restore(sess, './model.ckpt')
            data = data.reshape(1, self.input_dim[0], self.input_dim[1], self.input_dim[2])
            output = sess.run(self.output_value, feed_dict={self.x: data})
        return output



