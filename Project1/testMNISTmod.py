import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
image_size = 784
n_class = 10

class Network(object):
    def __init__(self, lr, n_layers, n_hidden_neurons):
        self.lr = lr 
        self.n_layers = n_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.w = [None] * n_layers
        self.b = [None] * n_layers
        self.h = [None] * n_layers
        self.size = [None] * n_layers
        self.build_multilayer_graph()

    def build_multilayer_graph(self):
        self.x = tf.placeholder(tf.float32, [None, image_size])
        # the input layer
        self.h[0] = self.x
        self.size[0] = image_size
        # the hidden layers
        for l in range(1, self.n_layers - 1):
            self.size[l] = self.n_hidden_neurons
            self.w[l] = tf.Variable(tf.truncated_normal([self.size[l-1], self.size[l]], stddev=0.1))
            self.b[l] = tf.Variable(tf.zeros([self.size[l]]))
            # self.h[l] = tf.add(tf.matmul(self.h[l-1], self.w[l]), self.b[l])
            self.h[l] = tf.nn.relu(tf.add(tf.matmul(self.h[l-1], self.w[l]), self.b[l]))
        # the output layer
        self.size[-1] = n_class
        self.w[-1] = tf.Variable(tf.truncated_normal([self.size[-2], self.size[-1]], stddev=0.1))
        self.b[-1] = tf.Variable(tf.zeros([self.size[-1]]))
        self.y = tf.add(tf.matmul(self.h[-2], self.w[-1]), self.b[-1])
        self.y_ = tf.placeholder(tf.float32, [None, n_class])
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cross_entropy)

    def train(self):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for _ in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
        
    def eval(self):
        self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        return self.sess.run(accuracy, feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels})

if __name__ == "__main__":
    # test different learning rate
    for i in range(10):
        lr = (i + 1) * 0.1
        n_layers = 3
        n_hidden_neurons = int((image_size * n_class) ** 0.5) 
        p = Network(lr, n_layers, n_hidden_neurons)
        p.train()
        accuracy = p.eval()
        print('{}, {}, {}: {}'.format(lr, n_layers, n_hidden_neurons, accuracy))

    # test different number of hidden layers 
    for i in range(5):
        lr = 0.1
        n_layers = 2 + i
        n_hidden_neurons = int((image_size * n_class) ** 0.5) 
        p = Network(lr, n_layers, n_hidden_neurons)
        p.train()
        accuracy = p.eval()
        print('{}, {}, {}: {}'.format(lr, n_layers, n_hidden_neurons, accuracy))

    # test different number of neurons in hidden layers
    for i in range(11):
        lr = 0.1
        n_layers = 3
        n_hidden_neurons = int(image_size * 0.1 * i) + n_class
        p = Network(lr, n_layers, n_hidden_neurons)
        p.train()
        accuracy = p.eval()
        print('{}, {}, {}: {}'.format(lr, n_layers, n_hidden_neurons, accuracy))
    '''
    lr = 1.0
    n_layers = 3
    n_hidden_neurons = 50
    p = Network(lr, n_layers, n_hidden_neurons)
    p.train()
    accuracy = p.eval()
    print('{}, {}, {}: {}'.format(lr, n_layers, n_hidden_neurons, accuracy))
    '''
