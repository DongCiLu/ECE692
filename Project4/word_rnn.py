import numpy as np
from sklearn.datasets import fetch_20newsgroups
import unicodedata
import gensim
import string
import re
import collections
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import sys

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell

def load_dataset(dataset_fn):
    #load dataset
    print("loading dataset")
    with open(dataset_fn,'r') as f:
        text = f.read()
        
    #clean up text
    text = text.replace("\n"," ") #remove linebreaks
    text = re.sub(' +',' ',text) #remove duplicate spaces
    text = text.lower() #lowercase  
    
    # convert text to list of sentences
    print("converting text to list of sentences")
    sentences = re.sub(r'-|\t|\n',' ',text)
    sentences = sentences.split('.')
    remove_table = str.maketrans("","",string.punctuation)
    sentences = [sentence.translate(remove_table).lower().split() for sentence in sentences]
    
    # combine list of sentences to list of words
    # words = re.sub(r'-|\t|\n',' ',text)
    # words = words.split(" ")
    # words = re.sub("[^\w]", " ",  text).split()
    words = sum(sentences, [])
    
    return text, sentences, words
        
def plot_word_embeddings(sentences, model, wv_dim):
    #get most common words
    print("getting common words")
    dataset = [item for sublist in sentences for item in sublist]
    counts = collections.Counter(dataset).most_common(500)

    #reduce embeddings to 2d using tsne
    print("reducing embeddings to 2D")
    embeddings = np.empty((500,wv_dim))
    for i in range(500):
        embeddings[i,:] = model[counts[i][0]]
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=7500)
    embeddings = tsne.fit_transform(embeddings)

    #plot embeddings
    print("plotting most common words")
    fig, ax = plt.subplots(figsize=(30, 30))
    for i in range(500):
        ax.scatter(embeddings[i,0],embeddings[i,1])
        ax.annotate(counts[i][0], (embeddings[i,0],embeddings[i,1]))

    #save to disk
    plt.savefig('plot.png')

def train_word_embeddings(sentences, wv_dim, plot_flag = False):
    # train word2vec
    print("training word2vec")
    model = gensim.models.Word2Vec(sentences, min_count=1, size=wv_dim, workers=4)
    
    if plot_flag:
        plot_word_embeddings(sentences, model, wv_dim)
    
    return model
    
class vector_rnn(object):
    '''
    sample character-level RNN by Shang Gao
    
    parameters:
      - seq_len: integer (default: 200)
        number of characters in input sequence
      - first_read: integer (default: 50)
        number of characters to first read before attempting to predict next character
      - rnn_size: integer (default: 200)
        number of rnn cells
       
    methods:
      - train(text,iterations=100000)
        train network on given text
    '''
    def __init__(self,wv_model,wv_dim,seq_len=200,first_read=50,rnn_size=200):
        self.seq_len = seq_len
        self.first_read = first_read
    
        #dictionary mapping words to indices
        self.num_words = len(wv_model.wv.vocab)      
        print("number of words ", self.num_words)
        embedding_matrix = np.zeros((self.num_words, wv_dim))
        for i in range(len(wv_model.wv.vocab)):
            embedding_vector = wv_model.wv[wv_model.wv.index2word[i]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        # memory efficient way to load the embeddings(avoids several copies of embeddings) in tf
        #embedding layer weights are frozen to avoid updating embeddings while training
        embedding_weights = tf.Variable(
                tf.constant(0.0, shape=[self.num_words, wv_dim]),
                trainable=False, name="embedding_weights") 
                
        embedding_placeholder = tf.placeholder(
                tf.float32, [self.num_words, wv_dim])
        embedding_init = embedding_weights.assign(embedding_placeholder)
        
        self.word2idx = {word:idx for (idx,word) in enumerate(wv_model.wv.index2word)}
        self.idx2word = {idx:word for (idx,word) in enumerate(wv_model.wv.index2word)}

        '''
        #training portion of language model
        '''

        #input sequence of character indices
        self.words = tf.placeholder(tf.int32,[1,self.seq_len])
        print("shape of words: ", self.words.get_shape())
        embeddings = tf.nn.embedding_lookup(embedding_matrix, self.words)
        print("shape of embeddings: ", embeddings.get_shape())
        
        #rnn layer
        self.gru = GRUCell(rnn_size)
        outputs,states = tf.nn.dynamic_rnn(
                self.gru,embeddings,sequence_length=[seq_len],dtype=tf.float64)
        # outputs = tf.squeeze(outputs,[0])
        print("shape of states: ", states.get_shape())
        print("shape of outputs: ", outputs.get_shape())

        # #ignore all outputs during first read steps
        # outputs = outputs[:,first_read:-1]
        
        #softmax logit to predict next character (actual softmax is applied in cross entropy function)
        logits = tf.layers.dense(outputs,self.num_words,
                None,True,tf.orthogonal_initializer(),name='dense')
        # logits = tf.layers.dense(outputs,wv_dim,
                # None,True,tf.orthogonal_initializer(),name='dense')
        print("shape of logits: ", logits.get_shape())        
        # softmax_w = tf.get_variable(
                # "softmax_w", [rnn_size, self.num_words], dtype=tf.float64)
        # softmax_b = tf.get_variable("softmax_b", [self.num_words], dtype=tf.float64)
        # logits2 = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
        # print("shape of alternative logits: ", logits2.get_shape())   

        #target character at each step (after first read chars) is following character        
        targets = tf.one_hot(self.words, self.num_words)
        print("shape of targets: ", targets.get_shape()) 
        
        #loss and train functions
        self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=targets))
        self.optimizer = tf.train.AdamOptimizer(0.0002,0.9,0.999).minimize(self.loss)
        
        '''
        #generation portion of language model
        '''
        
        #use output and state from last word in training sequence
        state = tf.expand_dims(states[-1],0)
        output = embeddings[:,-1]
        print("shape of state: ", state.get_shape())
        print("shape of output: ", output.get_shape())
        
        #save predicted characters to list
        self.predictions = []
        
        #generate 100 new characters that come after input sequence
        for i in range(100):
        
            #run GRU cell and softmax 
            output,state = self.gru(output,state)
            logits = tf.layers.dense(output,self.num_words,None,True,tf.orthogonal_initializer(),name='dense',reuse=True)
            
            #get index of most probable character
            output = tf.argmax(tf.nn.softmax(logits),1)

            #save predicted character to list
            self.predictions.append(output)
            
            #one hot and cast to float for GRU API
            output = tf.nn.embedding_lookup(embedding_matrix, output)
            # output = tf.cast(tf.one_hot(output,self.num_words),tf.float32)
        
        #init op
        self.sess = tf.Session()
        self.sess.run(embedding_init, 
                feed_dict={embedding_placeholder: embedding_matrix})
        self.sess.run(tf.global_variables_initializer())

    def train(self,words,iterations=10000):
        '''
        train network on given text
                
        parameters:
          - text: string
            string to train network on
          - iterations: int (default: 100000)
            number of iterations to train for
        
        outputs:
            None
        '''
        
        #convert characters to indices
        print("converting text in indices")
        text_indices = [self.word2idx[word] for word in words if word in self.word2idx]
        
        #get length of text
        text_len = len(text_indices)
        
        #train
        for i in range(iterations):
        
            #select random starting point in text
            start = np.random.randint(text_len - self.seq_len)
            sequence = text_indices[start:start+self.seq_len]
            
            #train
            feed_dict = {self.words:[sequence]}
            loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)
            if (i+1) % 100 == 0:
                sys.stdout.write("iterations %i loss: %f  \r" % (i+1,loss))
                sys.stdout.flush()
            
            #show generated sample every 100 iterations
            if (i+1) % 100 == 0:
            
                feed_dict = {self.words:[sequence]}
                pred = self.sess.run(self.predictions,feed_dict=feed_dict)
                sample = ' '.join([self.idx2word[idx[0]] for idx in pred])
                print("iteration %i generated sample: %s" % (i+1,sample))
    
if __name__ == "__main__":
    dataset_fn = 'tcomc.txt'
    wv_dim = 50
    
    text, sentences, words = load_dataset(dataset_fn)
    print("text size ", len(text))
    print(text[:1000])
    print("sentence size ", len(sentences))
    print(sentences[1], sentences[-2])
    print("word size ", len(words))
    print(words[:200])
    
    wv_model = train_word_embeddings(sentences, wv_dim)
    print(wv_model.wv['man'])
    print(wv_model.wv['woman'])
    print(wv_model.wv['girl'])
    print(wv_model.wv.similarity('woman', 'man'), wv_model.wv.similarity('woman', 'green'))
    
    rnn = vector_rnn(wv_model, wv_dim)
    rnn.train(words, iterations=100000)
    