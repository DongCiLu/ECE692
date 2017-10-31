import numpy as np
import tensorflow as tf

# from yadlt.models.autoencoders import denoising_autoencoder
# from yadlt.models.convolutional import conv_net
from yadlt.utils import datasets, utilities
import conv_net

def run_cnn(dataset_dir):
    # utilities.random_seed_np_tf(FLAGS.seed)
    utilities.random_seed_np_tf(-1)
    
    # common parameters
    cifar_dir = dataset_dir
    num_epochs = 3
    batch_size = 64
    n_classes = 10
    
    # parameters for cnn
    name_cnn = 'cnn'
    original_shape_cnn = '32,32,3'
    layers_cnn = 'conv2d-5-5-32-1,maxpool-2,conv2d-5-5-64-1,maxpool-2,full-1024,softmax'
    loss_func_cnn = 'softmax_cross_entropy'
    opt_cnn = 'adam'
    learning_rate_cnn = 1e-4
    momentum_cnn = 0.5 # not used
    dropout_cnn = 0.5
    batch_norm = True
    
    # prepare data
    trX, trY, teX, teY = datasets.load_cifar10_dataset(cifar_dir, mode='supervised')
    # due to the memory limit, cannot use the whole training set
    trY_non_one_hot = trY
    trY = np.array(utilities.to_one_hot(trY))
    teY = np.array(teY)
    teY_non_one_hot = teY[5000:]
    teY = np.array(utilities.to_one_hot(teY))
    # first half test set is validation set
    vlX = teX[:5000]
    vlY = teY[:5000]
    teX = teX[5000:]
    teY = teY[5000:]
    
    # define Convolutional Network
    cnn = conv_net.ConvolutionalNetwork(
        original_shape=[int(i) for i in original_shape_cnn.split(',')],
        layers=layers_cnn, name=name_cnn, loss_func=loss_func_cnn,
        num_epochs=num_epochs, batch_size=batch_size, opt=opt_cnn,
        learning_rate=learning_rate_cnn, momentum=momentum_cnn, dropout=dropout_cnn,
        batch_norm = batch_norm
    )
    
    print('Start Convolutional Network training...')
    cnn.fit(trX, trY, vlX, vlY)  # supervised learning
    
if __name__ == '__main__':
    dataset_dir = 'cifar10'
    run_cnn(dataset_dir)
