import numpy as np
import tensorflow as tf

# from yadlt.models.autoencoders import denoising_autoencoder
# from yadlt.models.convolutional import conv_net
from yadlt.utils import datasets, utilities
import denoising_autoencoder
import conv_net


if __name__ == '__main__':
    # utilities.random_seed_np_tf(FLAGS.seed)
    utilities.random_seed_np_tf(-1)
    
    # common parameters
    cifar_dir = 'cifar10'
    num_epochs = 3
    batch_size = 64
    n_classes = 10
    
    # parameters for dae
    name_dae = 'dae'
    n_components_dae = 1024
    enc_act_func_dae = tf.nn.sigmoid
    dec_act_func_dae = tf.nn.sigmoid
    corr_type_dae = 'masking'
    corr_frac_dae = 0.5
    loss_func_dae = 'cross_entropy'
    opt_dae = 'momentum'
    regcoef_dae = 5e-4
    learning_rate_dae = 0.05
    momentum_dae = 0.9
    
    # parameters for cnn
    name_cnn = 'cnn'
    original_shape_cnn = '32,32,3'
    layers_cnn = 'conv2d-5-5-32-1,maxpool-2,conv2d-5-5-64-1,maxpool-2,full-1024,softmax'
    loss_func_cnn = 'softmax_cross_entropy'
    opt_cnn = 'adam'
    learning_rate_cnn = 1e-4
    momentum_cnn = 0.5 # not used
    dropout_cnn = 0.5
    
    # loading data
    trX, trY, teX, teY = datasets.load_cifar10_dataset(cifar_dir, mode='supervised')
    trY = np.array(utilities.to_one_hot(trY))
    teY = np.array(teY)
    teY = np.array(utilities.to_one_hot(teY))
    vlX = teX[:5000]
    vlY = teY[:5000]
    # n_labels_tr = trY.shape[0]
    # index_offset_tr = np.arange(n_labels_tr) * n_classes
    # trY_one_hot = np.zeros((n_labels_tr, n_classes))
    # trY_one_hot.flat[index_offset_tr + trY.ravel()] = 1
    # teY = np.array(teY)
    # n_labels_te = teY.shape[0]
    # index_offset_te = np.arange(n_labels_te) * n_classes
    # teY_one_hot = np.zeros((n_labels_te, n_classes))
    # teY_one_hot.flat[index_offset_te + teY.ravel()] = 1
    # vlX = teX[:5000]
    # vlY_one_hot = teY_one_hot[:5000]
    
    # define Denoising Autoencoder
    # dae = denoising_autoencoder.DenoisingAutoencoder(
        # name=name_dae, n_components=n_components_dae,
        # enc_act_func=enc_act_func_dae, dec_act_func=dec_act_func_dae,
        # corr_type=corr_type_dae, corr_frac=corr_frac_dae,
        # loss_func=loss_func_dae, opt=opt_dae, regcoef=regcoef_dae,
        # learning_rate=learning_rate_dae, momentum=momentum_dae,
        # num_epochs=num_epochs, batch_size=batch_size
    # )

    # print('Start Denoising Autoencoder training...')
    # dae.fit(trX, trX, vlX, vlX) # unsupervised learning
    
    # define Convolutional Network
    cnn = conv_net.ConvolutionalNetwork(
        original_shape=[int(i) for i in original_shape_cnn.split(',')],
        layers=layers_cnn, name=name_cnn, loss_func=loss_func_cnn,
        num_epochs=num_epochs, batch_size=batch_size, opt=opt_cnn,
        learning_rate=learning_rate_cnn, momentum=momentum_cnn, dropout=dropout_cnn
    )
    
    print('Start Convolutional Network training...')
    cnn.fit(trX, trY, vlX, vlY)  # supervised learning
    # cnn.fit(trX, trY_one_hot, vlX, vlY_one_hot)  # supervised learning