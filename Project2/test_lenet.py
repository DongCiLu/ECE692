import tensorflow as tf
from datasets import cifar10
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

with tf.Graph().as_default():
    train_dir = 'tensorflow_log_lenet'
    data_dirname = 'cifar10'

    batch_size = 128

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
            'eval/Recall@5': slim.metrics.streaming_recal_at_k(\
            logits, labels, 5)})

    print 'Running evaluation loop ...'
    checkpoint_path = tf.train.latest_checkpint(train_dir)
    metric_values = slim.evalution.evaluate_once(\
            master = '', \
            checkpoint_path = checkpoint_path, \
            logdir = train_dir, \
            eval_op = names_to_updates.values(), \
            final_op = names_to_values.values())

    names_to_values = dict(zip(name_to_values.keys(), metric_values))
    for name in names_to_values:
        print '{}: {}'.format(name, names_to_values[name])

