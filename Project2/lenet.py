import tensorflow as tf
from datasets import cifar10
import numpy as np

slim = tf.contrib.slim

def load_batch(dataset, batch_size, epochs):
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    [image, label] = provider.get(['image', 'label'])
    images, labels = tf.train.batch(\
            [image, label], \
            batch_size = batch_size, \
            capacity = epochs * batch_size)
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

if __name__ == '__main__':
    train_dir = './tensorflow_log_lenet/'
    data_dirname = './datasets'
    lr = 0.01
    epochs = 1000
    batch_size = 128

    # load training data
    dataset = cifar10.get_split('train', data_dirname)
    images, labels = load_batch(dataset, \
            batch_size = batch_size, epochs = epochs)

    # define the loss
    logits, end_points = leNet(images, \
            n_class = dataset.num_classes, \
            is_training = True)
    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
    slim.losses.softmax_cross_entropy(logits, one_hot_labels)
    total_loss = slim.losses.get_total_loss()

    # monitor the total loss
    tf.summary.scalar('losses/Total Loss', total_loss)

    # specify the optimizer and create the train op
    optimizer = tf.train.AdamOptimizer(learning_rate = lr)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    # run the training
    final_loss = slim.learning.train( \
            train_op, \
            logdir = train_dir, \
            number_of_steps = epochs, \
            save_summaries_secs = 1)

    print 'Finished_traing. Final batch loss {}'.format(final_loss)

