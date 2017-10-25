import numpy as np
import pickle
import tensorflow as tf
import argparse

# from yadlt.models.autoencoders import denoising_autoencoder
# from yadlt.models.convolutional import conv_net
from yadlt.utils import datasets, utilities
import denoising_autoencoder
import conv_net

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def generate_feature_sets(dataset_dir, fs_filename, tr_size):
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
    # due to the memory limit, cannot use the whole training set
    trX = trX[:tr_size]
    trY = trY[:tr_size]
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
    fs_file = open(fs_filename, 'wb')
    pickle.dump(trY_non_one_hot, fs_file)
    pickle.dump(teY_non_one_hot, fs_file)
    
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
    
    feature_train_set_1 = dae.extract_features(trX)
    pickle.dump(feature_train_set_1, fs_file)
    feature_test_set_1 = dae.extract_features(teX)
    pickle.dump(feature_test_set_1, fs_file)
    
    # define Convolutional Network
    cnn = conv_net.ConvolutionalNetwork(
        original_shape=[int(i) for i in original_shape_cnn.split(',')],
        layers=layers_cnn, name=name_cnn, loss_func=loss_func_cnn,
        num_epochs=num_epochs, batch_size=batch_size, opt=opt_cnn,
        learning_rate=learning_rate_cnn, momentum=momentum_cnn, dropout=dropout_cnn
    )
    
    print('Start Convolutional Network training...')
    cnn.fit(trX, trY, vlX, vlY)  # supervised learning
    
    feature_train_set_2 = cnn.extract_features(trX)
    pickle.dump(feature_train_set_2,  fs_file)
    feature_test_set_2 = cnn.extract_features(teX)
    pickle.dump(feature_test_set_2,  fs_file)
    fs_file.close()
    
def load_feature_sets(fs_filename):
    fs_file = open(fs_filename, 'rb')
    train_labels = pickle.load(fs_file)
    test_labels = pickle.load(fs_file)
    feature_train_set_1 = pickle.load(fs_file)
    feature_test_set_1 = pickle.load(fs_file)
    feature_train_set_2 = pickle.load(fs_file)
    feature_test_set_2 = pickle.load(fs_file)
    fs_file.close()
    
    return train_labels, test_labels, feature_train_set_1, \
            feature_test_set_1, feature_train_set_2, feature_test_set_2
            
def classify_with_svm(train_labels, test_labels, feature_train_set_1, \
            feature_test_set_1, feature_train_set_2, feature_test_set_2) :
        
        svm1 = SVC()
        svm2 = SVC()
        
        saved_train_size = train_labels.shape[0]
        total_step = 10
        
        for step in range(total_step):
            batch_train_size = int(saved_train_size / total_step * (step + 1))
            
            svm1.fit(feature_train_set_1[:batch_train_size], 
                    train_labels[:batch_train_size])
            svm2.fit(feature_train_set_2[:batch_train_size], 
                    train_labels[:batch_train_size])
            
            test_result_1 = svm1.predict(feature_test_set_1)
            test_result_2 = svm2.predict(feature_test_set_2)
            
            accuracy1 = accuracy_score(test_labels, test_result_1)
            accuracy2 = accuracy_score(test_labels, test_result_2)
            
            print("{}, {}".format(accuracy1, accuracy2))

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('mode', type=str)
    arg_parser.add_argument('--tr_size', type=int, default=10000, required=False)
    args = arg_parser.parse_args()
    
    assert args.mode in ['generate', 'classify']
    
    dataset_dir = 'cifar10'
    fs_filename = 'feature_sets.pkl'
    
    if args.mode == 'generate':
        generate_feature_sets(dataset_dir, fs_filename, args.tr_size)
        
    elif args.mode == 'classify':
        train_labels, test_labels, feature_train_set_1, \
                feature_test_set_1, feature_train_set_2, feature_test_set_2 \
                = load_feature_sets(fs_filename)
        
        classify_with_svm(train_labels, test_labels, feature_train_set_1, \
                feature_test_set_1, feature_train_set_2, feature_test_set_2)
        
        


    