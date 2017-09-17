# Sample code implementing LeNet-5 from Liu Liu
import cPickle

import tensorflow as tf
import numpy as np

def unpickle(filename):
    with open(filename, 'rb') as f:
        data = cPickle.load(f)
    return data

class CNN(object):
    def __init__(self, log_dirname, lr, epochs, \
            batch_size, input_size, n_class):
        self.log_dirname = log_dirname

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.isize_dim = input_size
        self.isize = input_size[0] * input_size[1] * input_size[2]
        self.osize = n_class

        self.kernal = [5, 5]
        self.feature = [6, 16]
        self.fc = [120, 84]

        self.build_graph()

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.isize])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.osize])
        
        # define conv-layer variables
        W_conv1 = self.weight_variable(\
                [self.kernal[0], self.kernal[0], \
                self.isize_dim[2], self.feature[0]])    
        b_conv1 = self.bias_variable([self.feature[0]])
        W_conv2 = self.weight_variable(\
                [self.kernal[1], self.kernal[1], \
                self.feature[0], self.feature[1]])
        b_conv2 = self.bias_variable([self.feature[1]])
        
        x_image = tf.reshape(self.x, [-1, self.isize_dim[0], \
                self.isize_dim[1], self.isize_dim[2]])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # densely/fully connected layer
        fmap_size = self.isize_dim[0] / (2 ** len(self.feature))
        W_fc1 = self.weight_variable([fmap_size * fmap_size * 
            self.feature[1], self.fc[0]])
        b_fc1 = self.bias_variable([self.fc[0]])

        h_pool2_flat = tf.reshape(h_pool2, \
                [-1, fmap_size * fmap_size * self.feature[1]])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout regularization
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = self.weight_variable([self.fc[0], self.fc[1]])
        b_fc2 = self.bias_variable([self.fc[1]])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        # dropout regularization
        h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)

        # linear classifier
        W_fc3 = self.weight_variable([self.fc[1], self.osize])
        b_fc3 = self.bias_variable([self.osize])

        self.y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
        cross_entropy = tf.reduce_mean(\
                tf.nn.softmax_cross_entropy_with_logits(\
                labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(\
                self.lr).minimize(cross_entropy)

        self.writer = tf.summary.FileWriter(\
                self.log_dirname, tf.get_default_graph())


    def train(self, x_train, y_train):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.eval() # creating evaluation
        train_size = x_train.shape[0]
        for i in range(self.epochs):
            start = (i * self.batch_size) % train_size
            if start + self.batch_size > train_size:
                continue
            x_batch = x_train[start: start + self.batch_size]
            y_batch = y_train[start: start + self.batch_size]
            if i % 100 == 0:
                train_acc = self.sess.run(self.accuracy, \
                        feed_dict={self.x: x_batch, self.y_: y_batch, \
                        self.keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_acc))
            self.sess.run([self.train_step], \
                    feed_dict={self.x: x_batch, self.y_: y_batch, \
                    self.keep_prob: 0.5})
        
    def eval(self):
        correct_prediction = tf.equal(\
                tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(\
                tf.cast(correct_prediction, tf.float32))

    def test_eval(self, x_test, y_test):
        self.eval()
        test_acc = self.sess.run(self.accuracy, feed_dict={
                self.x: x_test, self.y_: y_test, self.keep_prob: 1.0})
        print('test accuracy %g' % test_acc)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':
    log_dirname = 'tensorflow_log_lenet/'
    # load training data
    data_train = np.empty([0, 3072])
    single_label_train = np.empty([0, 1])
    for i in range(1, 6):
        filename = 'cifar-10-batches-py/data_batch_{}'.format(i)
        rawdata = unpickle(filename)
        data_i = rawdata['data']
        single_label_i = np.reshape(np.array(rawdata['labels']), \
                [data_i.shape[0], 1])
        data_train = np.concatenate((data_train, data_i))
        single_label_train = np.concatenate(\
                (single_label_train, single_label_i))
    # load testing data
    filename = 'cifar-10-batches-py/test_batch'
    rawdata = unpickle(filename)
    data_test = rawdata['data']
    single_label_test = np.reshape(np.array(rawdata['labels']), \
            [data_test.shape[0], 1])

    label_train = []
    for item in single_label_train:
        temp_label = [0] * 10
        temp_label[item[0]] = 1
        label_train.append(temp_label)
    label_train = np.array(label_train)

    label_test = []
    for item in single_label_test:
        temp_label = [0] * 10
        temp_label[item[0]] = 1
        label_test.append(temp_label)
    label_test = np.array(label_test)

    # split = int(data.shape[0] * 0.8)
    # x_train, y_train = data[:split], label[:split]
    # x_valid, y_valid = data[split:], label[split:]
    x_train, y_train = data_train, label_train
    x_test, y_test = data_test, label_test

    lr = 1e-3
    epochs = 1000
    batch_size = 128
    input_size = [32, 32, 3]
    n_class = 10

    cnn = CNN(log_dirname, lr, epochs, batch_size, input_size, n_class)
    cnn.train(x_train, y_train)
    cnn.test_eval(x_test, y_test)
