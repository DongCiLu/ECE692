import numpy as np
import pickle
import tensorflow as tf
import argparse

# from yadlt.models.autoencoders import denoising_autoencoder
# from yadlt.models.convolutional import conv_net
from yadlt.utils import datasets, utilities
import denoising_autoencoder
import conv_net

# import sklearn
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib

def generate_feature_sets(dataset_dir, fs1_filename, fs2_filename):
    # utilities.random_seed_np_tf(FLAGS.seed)
    utilities.random_seed_np_tf(-1)
    
    # common parameters
    cifar_dir = dataset_dir
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
    teY_non_one_hot = teY[5000:]
    teY = np.array(utilities.to_one_hot(teY))
    # first half test set is validation set
    vlX = teX[:5000]
    vlY = teY[:5000]
    teX = teX[5000:]
    teY = teY[5000:]
    
    # define Denoising Autoencoder
    dae = denoising_autoencoder.DenoisingAutoencoder(
        name=name_dae, n_components=n_components_dae,
        enc_act_func=enc_act_func_dae, dec_act_func=dec_act_func_dae,
        corr_type=corr_type_dae, corr_frac=corr_frac_dae,
        loss_func=loss_func_dae, opt=opt_dae, regcoef=regcoef_dae,
        learning_rate=learning_rate_dae, momentum=momentum_dae,
        num_epochs=num_epochs, batch_size=batch_size
    )

    print('Start Denoising Autoencoder training...')
    dae.fit(trX, trX, vlX, vlX) # unsupervised learning
    
    feature_set_1 = dae.extract_features(teX)
    fs1_file = open(fs1_filename, 'wb')
    pickle.dump(feature_set_1, fs1_file)
    pickle.dump(teY_non_one_hot, fs1_file)
    fs1_file.close()
    
    # define Convolutional Network
    cnn = conv_net.ConvolutionalNetwork(
        original_shape=[int(i) for i in original_shape_cnn.split(',')],
        layers=layers_cnn, name=name_cnn, loss_func=loss_func_cnn,
        num_epochs=num_epochs, batch_size=batch_size, opt=opt_cnn,
        learning_rate=learning_rate_cnn, momentum=momentum_cnn, dropout=dropout_cnn
    )
    
    print('Start Convolutional Network training...')
    cnn.fit(trX, trY, vlX, vlY)  # supervised learning
    
    feature_set_2 = cnn.extract_features(teX)
    fs2_file = open(fs2_filename, 'wb')
    pickle.dump(feature_set_2,  fs2_file)
    pickle.dump(teY_non_one_hot, fs2_file)
    fs1_file.close()
    
def load_feature_sets(fs1_filename, fs2_filename):
    fs1_file = open(fs1_filename, 'rb')
    feature_set_1 = pickle.load(fs1_file)
    fs1_file.close()
    
    fs2_file = open(fs2_filename, 'rb')
    feature_set_2 = pickle.load(fs2_file)
    labels = pickle.load(fs2_file)
    fs2_file.close()
    
    return feature_set_1, feature_set_2, labels

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('mode', type=str)
    args = arg_parser.parse_args()
    
    assert args.mode in ['generate', 'classify']
    
    dataset_dir = 'cifar10'
    fs1_filename = 'feature_set_1.pkl'
    fs2_filename = 'feature_set_2.pkl'
    
    if args.mode == 'generate':
        generate_feature_sets(dataset_dir, fs1_filename, fs2_filename)
        
    elif args.mode == 'classify':
        feature_set_1, feature_set_2, labels = \
                load_feature_sets(fs1_filename, fs2_filename)
        print(feature_set_1.shape)
        print(feature_set_2.shape)
        print(labels.shape)
        
        svm = SVC()
        


    