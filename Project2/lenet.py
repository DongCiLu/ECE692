import tensorflow as tf
from datasets import cifar10
import numpy as np
from tensorflow.contrib.slim.python.slim.learning import train_step

slim = tf.contrib.slim

def load_batch(dataset, batch_size):
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    [image, label] = provider.get(['image', 'label'])
    images, labels = tf.train.batch(\
            [image, label], \
            batch_size = batch_size, \
            capacity = 2 * batch_size)
    images = tf.to_float(images)
    return images, labels

def leNet(images, n_class=10, \
        is_training=False, scope='LeNet'):

    kernal = [5, 5]
    feature = [6, 16]
    pool = [2, 2]
    fc = [1024]
    dropout_keep_prob = 0.5

    end_points = {}
    with tf.variable_scope(scope, 'LeNet', [images, n_class]):
        # first convolution layer
        net = slim.conv2d(images, feature[0], \
                [kernal[0], kernal[0]], scope='conv1')
        net = slim.max_pool2d(net, \
                [pool[0], pool[0]], scope='pool1')
        # second convolution layer
        net = slim.conv2d(net, feature[1], \
                [kernal[1], kernal[1]], scope='conv2')
        net = slim.max_pool2d(net, \
                [pool[1], pool[1]], scope='pool2')
        # flatten output from convolutional layer
        net = slim.flatten(net)
        end_points['Flatten'] = net
        # first fully connected layer
        net = slim.fully_connected(net, \
                fc[0], scope='fc3')
        # dropout regulation for fully connected layer
        dropuot_keep_prob = 0.5
        net = slim.dropout(net, dropout_keep_prob, \
                is_training=is_training, scope='dropout3')
        # output fully connected layer
        logits = slim.fully_connected(net, n_class, \
                activation_fn=None, scope='fc4')

        end_points['Logits'] = logits 
        predictions = tf.argmax(logits, 1)
        end_points['Predictions'] = predictions

        return logits, end_points

def train_step_fn(session, *args, **kwargs):
    total_loss, should_stop = train_step(session, *args, **kwargs)

    if train_step_fn.step % train_step_fn.test_step == 0:
        session.run(train_step_fn.accuracy_test)

    train_step_fn.step += 1

    return [total_loss, should_stop]

if __name__ == '__main__':
    train_dir = 'tensorflow_log_lenet'
    data_dirname = 'cifar10'
    lr = 0.0001
    epochs = 1000
    batch_size = 128

    with tf.Graph().as_default():
        # load training data
        dataset_train = cifar10.get_split('train', data_dirname)
        images_train, labels_train = load_batch(dataset_train, \
                    batch_size = batch_size)
        dataset_test = cifar10.get_split('test', data_dirname)
        images_test, labels_test = load_batch(dataset_test, \
                batch_size = batch_size)

        # define the loss
        logits_train, end_points_train = leNet(images_train, \
                n_class = dataset.num_classes, \
                is_training = True)
        one_hot_labels_train = slim.one_hot_encoding(\
                labels_train, dataset.num_classes)
        slim.losses.softmax_cross_entropy(\
                logits_train, one_hot_labels_train)
        total_loss = slim.losses.get_total_loss()
        logits_test, end_points_test = leNet(images_test, \
                n_class = dataset.num_classes, \
                is_training = False)

        # specify the optimizer and create the train op
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        train_op = slim.learning.create_train_op(\
                total_loss, optimizer)

        accuracy_train = slim.metrics.accuracy(\
                    end_points_train['Predictions'], labels_train)
        accuracy_test = slim.metrics.accuracy(\
                    end_points_test['Predictions'], labels_test)

        # monitor the total loss
        tf.summary.scalar('accuracy/train_accuracy', accuracy_train)
        tf.summary.scalar('accuracy/test_accuracy', accuracy_test)

        train_step_fn.step = 0
        train_step_fn.test_step = 100
        train_step_fn.accuracy_test = accuracy_test

        # run the training
        final_loss = slim.learning.train( \
                train_op, \
                logdir = train_dir, \
                train_step_fn = train_step_fn, \
                number_of_steps = epochs, \
                save_summaries_secs = 1)
