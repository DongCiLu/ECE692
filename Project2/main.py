import tensorflow as tf
import cPickle

def unpickle(filename):
    with open(filename, 'rb') as f:
        data = cPickle.load(f)
    return data
