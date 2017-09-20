from datasets import cifar10

with tf.Graph().as_default():
    train_dir = 'tensorflow_log_lenet'
    data_dirname = 'cifar10'

    dataset = cifar10.get_slit('test', data_dirname)
    images, labels = load_batch(dataset, \
            batch_size = batch_size, epochs = epochs)

    logits, end_points = leNet(images, \
            n_class = dataset.num_classes, \
            is_training = True)
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
        print ('{}: {}'.format(name, names_to_values[name])

