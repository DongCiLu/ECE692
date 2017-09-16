import cPickle

import tensorflow as tf
import numpy as np

def unpickle(filename):
    with open(filename, 'rb') as f:
        data = cPickle.load(f)
    return data

class RESNET(object):
    def __init__(self, graph_filename, cp_filename):
        self.graph_filename = graph_filename
        self.cp_filename = cp_filename
        self.sess = tf.Session()

        self.restore_graph()

    def restore_graph(self):
        self.saver = tf.train.import_meta_graph(self.graph_filename)
        '''
        self.saver.restore(self.sess, \
                tf.train.latest_checkpoint(self.cp_filename))
        self.graph = tf.get_default_graph()
        '''

if __name__ == '__main__':
    filename = 'cifar-10-batches-py/data_batch_1'
    rawdata = unpickle(filename)
    data = rawdata['data']
    single_label = \
            np.reshape(np.array(rawdata['labels']), [data.shape[0], 1])

    label = []
    for item in single_label:
        temp_label = [0] * 10
        temp_label[item[0]] = 1
        label.append(temp_label)
    label = np.array(label)

    split = int(data.shape[0] * 0.8)
    x_train, y_train = data[:split], label[:split]
    x_valid, y_valid = data[split:], label[split:]

    graph_filename = \
            'tensorflow-resnet-pretrained-20160509/ResNet-L50.meta'
    cp_filename = \
            'tensorflow-resnet-pretrained-20160509/ResNet-L50.ckpt'


    lr = 1e-4
    epochs = 200
    batch_size = 100
    input_size = [32, 32, 3]
    n_class = 10

    resnet = RESNET(graph_filename, cp_filename)

    '''
    cnn = CNN(lr, epochs, batch_size, input_size, n_class)
    cnn.train(x_train, y_train)
    cnn.test_eval(x_valid, y_valid)
    '''
