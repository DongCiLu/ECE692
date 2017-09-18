import cPickle

import tensorflow as tf
import numpy as np

def unpickle(filename):
    with open(filename, 'rb') as f:
        data = cPickle.load(f)
    return data

class RESNET(object):
    def __init__(self, graph_filename, cp_filename, log_dirname):
        self.graph_filename = graph_filename
        self.cp_filename = cp_filename
        self.log_dirname = log_dirname
        self.sess = tf.Session()

        self.restore_graph()

    def restore_graph(self):
        self.saver = tf.train.import_meta_graph(self.graph_filename)
        self.saver.restore(self.sess, self.cp_filename)
        self.graph = tf.get_default_graph()
        self.writer = tf.summary.FileWriter(\
                self.log_dirname, graph = tf.get_default_graph())

    def train(self, x_train, y_train):
        return

    def eval(self, x_valid, y_valid):
        return

    def test(self, x_test, y_test):
        return

if __name__ == '__main__':
    # load training data
    data_train = np.empty([0, 3072], dtype = int)
    single_label_train = np.empty([0, 1], dtype = int)
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

    graph_filename = \
            'tensorflow-resnet-pretrained-20160509/ResNet-L50.meta'
    cp_filename = \
            'tensorflow-resnet-pretrained-20160509/ResNet-L50.ckpt'
    log_dirname = 'tensorflow_log_resnet/'


    lr = 1e-4
    epochs = 200
    batch_size = 100
    input_size = [32, 32, 3]
    n_class = 10

    resnet = RESNET(graph_filename, cp_filename, log_dirname)

    '''
    cnn = CNN(lr, epochs, batch_size, input_size, n_class)
    cnn.train(x_train, y_train)
    cnn.test_eval(x_valid, y_valid)
    '''
