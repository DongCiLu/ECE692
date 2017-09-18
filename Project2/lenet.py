import tensorflow as tf
from datasets import cifar10
import numpy as np

slim = tf.contrib.slim

class Lenet(object):
    def __init__(self, log_dirname, lr):
        self.log_dirname = log_dirname
        self.lr = lr

        self.kernal = [5, 5]
        self.feature = [6, 16]
        self.pool = [2, 2]
        self.fc = [1024]

    def network(self, images, n_class=10, \
            is_training=False, scope='LeNet'):
        end_points = {}
        with tf.variable_scope(scope, 'LeNet', [images, n_class]):
            # first convolution layer
            self.net = slim.conv2d(image, self.feature[0], \
                    [self.kernal[0], self.kernal[0]], scope='conv1')
            self.net = slim.max_pool2d(self.net, \
                    [self.pool[0], self.pool[0]], scope='pool1')
            # second convolution layer
            self.net = slim.conv2d(self.net, self.feature[1], \
                    [self.kernal[1], self.kernal[1]], scope='conv2')
            self.net = slim.max_pool2d(self.net, \
                    [self.pool[1], self.pool[1]], scope='pool2')
            # flatten output from convolutional layer
            self.net = slim.flatten(self.net)
            self.end_points['Flatten'] = self.net
            # first fully connected layer
            self.net = slim.fully_connected(self.net, \
                    self.fc[0], scope='fc3')
            # dropout regulation for fully connected layer
            dropuot_keep_prob = 0.5
            self.net = slim.dropout(net, dropout_keep_prob, \
                    is_training=is_training, scope='dropout3')
            # output fully connected layer
            logits = slim.fully_connected(self.net, n_class, \
                    activation_fn=None, scope='fc4')

            end_points['Logits'] = self.logits 
            predictions = tf.argmax(logits, 1)
            end_points['Predictions'] = predictions

            return predictions, end_points

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

if __name__ == '__main__':
    log_dirname = 'tensorflow_log_lenet/'
    data_dirname = './datasets'
    # load training data
    dataset = cifar10.get_split('validation', data_dirname)
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    [image, label] = provider.get(['image', 'label'])

    '''
    lr = 1e-3
    epochs = 1000
    batch_size = 128
    input_size = [32, 32, 3]
    n_class = 10

    cnn = CNN(log_dirname, lr, epochs, batch_size, input_size, n_class)
    cnn.train(x_train, y_train)
    cnn.test_eval(x_test, y_test)
    '''
