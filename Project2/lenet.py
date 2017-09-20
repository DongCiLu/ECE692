import tensorflow as tf
from datasets import cifar10
import numpy as np
import argparse

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

if __name__ == '__main__':
    train_dir = 'tensorflow_log_lenet'
    data_dirname = 'cifar10'
    lr = 0.01
    epochs = 1000
    batch_size = 128

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('mode', type=str)
    args = arg_parser.parse_args()
    mode = args.mode

    if mode == 'train':
        with tf.Graph().as_default():
            # load training data
            dataset = cifar10.get_split('train', data_dirname)
            images, labels = load_batch(dataset, \
                    batch_size = batch_size)

            # define the loss
            logits, end_points = leNet(images, \
                    n_class = dataset.num_classes, \
                    is_training = True)
            one_hot_labels = \
                    slim.one_hot_encoding(labels, dataset.num_classes)
            slim.losses.softmax_cross_entropy(logits, one_hot_labels)
            total_loss = slim.losses.get_total_loss()
            accuracy = slim.metrics.accuracy(\
                    end_points['Predictions'], labels)

            # monitor the total loss
            tf.summary.scalar('losses/Total Loss', total_loss)
            tf.summary.scalar('accuracy', accuracy)

            # specify the optimizer and create the train op
            optimizer = tf.train.AdamOptimizer(learning_rate = lr)
            train_op = slim.learning.create_train_op(\
                    total_loss, optimizer)

            # run the training
            final_loss = slim.learning.train( \
                    train_op, \
                    logdir = train_dir, \
                    number_of_steps = epochs, \
                    save_summaries_secs = 1)

            print 'Finished_traing. Final batch loss {}'.format(\
                    final_loss)
    elif mode == 'test':
        dataset = cifar10.get_split('test', data_dirname)
        images, labels = load_batch(dataset, \
                batch_size = batch_size)

        logits, end_points = leNet(images, \
                n_class = dataset.num_classes, \
                is_training = False)
        predictions = end_points['Predictions']

        names_to_values, names_to_updates = \
                slim.metrics.aggregate_metric_map({ \
                'eval/Accuracy': slim.metrics.streaming_accuracy(\
                predictions, labels), \
                'eval/Recall@5': slim.metrics.streaming_recall_at_k(\
                logits, labels, 5)})

        print 'Running evaluation loop ...'
        checkpoint_path = tf.train.latest_checkpoint(train_dir)
        metric_values = slim.evaluation.evaluate_loop(\
                master = '', \
                checkpoint_path = checkpoint_path, \
                logdir = train_dir, \
                eval_op = names_to_updates.values(), \
                final_op = names_to_values.values(), 
                eval_interval_secs = 10)

        names_to_values = \
                dict(zip(names_to_values.keys(), metric_values))
        for name in names_to_values:
            print '{}: {}'.format(name, names_to_values[name])

    else: 
        print 'Wrong running mode. Must be train or test.'

