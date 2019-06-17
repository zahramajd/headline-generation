import random
import codecs
import math
import time
import sys
import subprocess
import os.path
import pickle
import numpy as np
import gensim

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.wrappers import TimeDistributed
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers.recurrent import GRU
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.core import Lambda, Activation
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.preprocessing import sequence

from tqdm import tqdm
# from sklearn.cross_validation import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from numpy import inf
from operator import itemgetter

import parameters

seed = 28
random.seed(seed)
np.random.seed(seed)

empty_tag_location = 0
eos_tag_location = 1
unknown_tag_location = 2

word2vec = []
idx2word = {}
word2idx = {}
word2idx['<empty>'] = empty_tag_location
word2idx['<eos>'] = eos_tag_location
word2idx['<unk>'] = unknown_tag_location
idx2word[empty_tag_location] = '<empty>'
idx2word[eos_tag_location] = '<eos>'
idx2word[unknown_tag_location] = '<unk>'



def load_fasttext_embedding(file_name):

    idx = 3
    temp_word2vec_dict = {}

    temp_word2vec_dict['<empty>'] = [float(i) for i in np.random.rand(parameters.embedding_dimension, 1)]
    temp_word2vec_dict['<eos>'] = [float(i) for i in np.random.rand(parameters.embedding_dimension, 1)]
    temp_word2vec_dict['<unk>'] = [float(i) for i in np.random.rand(parameters.embedding_dimension, 1)]

    model = gensim.models.KeyedVectors.load_word2vec_format(file_name, limit = parameters.top_freq_word_to_use)
    V = model.index2word
    X = np.zeros((parameters.top_freq_word_to_use, model.vector_size))
    for index, word in enumerate(V):
        vector = model[word]
        temp_word2vec_dict[idx] = vector
        word2idx[word] = idx
        idx2word[idx] = word
        idx = idx + 1


    length_vocab = len(temp_word2vec_dict)
    shape = (length_vocab, parameters.embedding_dimension)
    word2vec = np.random.uniform(low=-1, high=1, size=shape)
    for i in range(length_vocab):
        if i in temp_word2vec_dict:
            word2vec[i, :] = temp_word2vec_dict[i]
            
    print('## embedding has been loaded ')
    return word2vec

def data_generator(number_words_to_replace, model, is_training=True, type='train'):
    
    def padding(list_idx, curr_max_length, is_left):
        if len(list_idx) >= curr_max_length:
            return list_idx
        number_of_empty_fill = curr_max_length - len(list_idx)
        if is_left:
            return [empty_tag_location, ] * number_of_empty_fill + list_idx
        else:
            return list_idx + [empty_tag_location, ] * number_of_empty_fill

    def headline2idx(list_idx, curr_max_length, is_input):

        if is_input:
            if len(list_idx) >= curr_max_length - 1:
                return list_idx[:curr_max_length - 1]
            else:
                list_idx = list_idx + [eos_tag_location, ]
                return padding(list_idx, curr_max_length - 1, False)
        else:
            if len(list_idx) == curr_max_length:
                list_idx[-1] = eos_tag_location
                return list_idx
            else:
                list_idx = list_idx + [eos_tag_location, ]
                return padding(list_idx, curr_max_length, False)

    def desc2idx(list_idx, curr_max_length):
        list_idx.reverse()
        list_idx = padding(list_idx, curr_max_length, True)
        list_idx = list_idx + [eos_tag_location, ]
        return list_idx

    def sentence2idx(sentence, is_headline, curr_max_length, is_input=True):

        list_idx = []
        tokens = sentence.split(" ")
        count = 0
        for each_token in tokens:
            if each_token in word2idx:
                list_idx.append(word2idx[each_token])
            else:
                list_idx.append(word2idx['<unk>'])
            count = count + 1
            if count >= curr_max_length:
                break

        if is_headline:
            return headline2idx(list_idx, curr_max_length, is_input)
        else:
            return desc2idx(list_idx, curr_max_length)

    def flip_headline(description_headline_data, number_words_to_replace, model):

        if number_words_to_replace <= 0 or model == None:
            return description_headline_data

        assert np.all(description_headline_data[:, parameters.max_len_desc] == eos_tag_location)

        batch_size = len(description_headline_data)
        predicated_headline_word_idx = model.predict(description_headline_data, verbose=1, batch_size = batch_size)
        copy_data = description_headline_data.copy()
        for idx in range(batch_size):

            random_flip_pos = sorted(random.sample(range(parameters.max_len_desc + 1, parameters.max_length), number_words_to_replace))
            for replace_idx in random_flip_pos:
                if (description_headline_data[idx, replace_idx] == empty_tag_location or
                description_headline_data[idx, replace_idx] == eos_tag_location):
                    continue

                new_id = replace_idx - (parameters.max_len_desc + 1)
                prob_words = predicated_headline_word_idx[idx, new_id]
                word_idx = prob_words.argmax()
                if word_idx == empty_tag_location or word_idx == eos_tag_location:
                    continue
                copy_data[idx, replace_idx] = word_idx
        return copy_data

    def convert_inputs(descriptions, headlines, number_words_to_replace, model, is_training):

        assert len(descriptions) == len(headlines)

        X, y = [], []
        for each_desc, each_headline in zip(descriptions, headlines):
            input_headline_idx = sentence2idx(each_headline, True, parameters.max_len_head, True)
            predicted_headline_idx = sentence2idx(each_headline, True, parameters.max_len_head, False)
            desc_idx = sentence2idx(each_desc, False, parameters.max_len_desc)

            assert len(input_headline_idx) == parameters.max_len_head - 1
            assert len(predicted_headline_idx) == parameters.max_len_head
            assert len(desc_idx) == parameters.max_len_desc + 1

            X.append(desc_idx + input_headline_idx)
            y.append(predicted_headline_idx)
            
        X, y = np.array(X), np.array(y)
        if is_training:
            X = flip_headline(X, number_words_to_replace, model)
            vocab_size = word2vec.shape[0]
            length_of_data = len(headlines)
            Y = np.zeros((length_of_data, parameters.max_len_head, vocab_size))
            for i, each_y in enumerate(y):
                Y[i, :, :] = np_utils.to_categorical(each_y, vocab_size)
            assert len(X)==len(Y)
            return X, Y
        else:
            return X,headlines

    with open(parameters.headlines_path, "r") as fp:
        headlines = fp.readlines()
        headlines = headlines
        dataLen = len(headlines)
        fp.close()

    with open(parameters.contents_path, "r") as fp:
        contents = fp.readlines()
        contents = contents
        fp.close()

    if type == 'train':
        idx_from = 0
        idx_to = int(parameters.train_val_percent * dataLen)

    if type == 'val':
        idx_from = int(parameters.train_val_percent * dataLen) + 1
        idx_to = dataLen - parameters.test_size

    if type == 'test':
        idx_from = dataLen - parameters.test_size + 1
        idx_to = dataLen -1


    headlines = headlines[idx_from:idx_to]
    contents = contents[idx_from:idx_to]


    while len(headlines) > 0:
        X, Y = [], []
        for i in range(128):
            if(len(headlines)<1): continue
            else:
                heads_line = headlines.pop()
                descs_line = contents.pop()
                X.append(descs_line)
                Y.append(heads_line)
        yield convert_inputs(X, Y, number_words_to_replace, model, is_training)

def simple_context(X, mask):

    desc, head = X[:, :parameters.max_len_desc, :], X[:, parameters.max_len_desc:, :]

    head_activations, head_words = head[:, :, :parameters.activation_rnn_size], head[:, :, parameters.activation_rnn_size:]
    desc_activations, desc_words = desc[:, :, :parameters.activation_rnn_size], desc[:, :, parameters.activation_rnn_size:]

    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2, 2))

    activation_energies = activation_energies + -1e20 * K.expand_dims(1. - K.cast(mask[:, :parameters.max_len_desc], 'float32'), 1)

    activation_energies = K.reshape(activation_energies, (-1, parameters.max_len_desc))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights, (-1, parameters.max_len_head, parameters.max_len_desc))

    desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2, 1))
    return K.concatenate((desc_avg_word, head_words))

def lstm_model():

        length_vocab, embedding_size = word2vec.shape

        model = Sequential()
        model.add(Embedding(length_vocab, embedding_size,
                        input_length=parameters.max_length,
                        weights=[word2vec], mask_zero=True,
                        name='embedding_layer'))

        for i in range(parameters.rnn_layers):
            lstm = LSTM(parameters.rnn_size, return_sequences=True, name='lstm_layer_%d' % (i + 1))
            model.add(lstm)

        model.add(Lambda(simple_context,
                     mask=lambda inputs, mask: mask[:, parameters.max_len_desc:],
                     output_shape = lambda input_shape: (input_shape[0], parameters.max_len_head, 2*(parameters.rnn_size - parameters.activation_rnn_size)),
                     name='simple_context_layer'))

        vocab_size = word2vec.shape[0]
        model.add(TimeDistributed(Dense(vocab_size, name='time_distributed_layer')))
        
        model.add(Activation('softmax', name='activation_layer'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        K.set_value(model.optimizer.lr, np.float32(parameters.learning_rate))
        print (model.summary())

        return model

def gru_model():
    
        length_vocab, embedding_size = word2vec.shape

        model = Sequential()
        model.add(Embedding(length_vocab, embedding_size,
                        input_length=parameters.max_length,
                        weights=[word2vec], mask_zero=True,
                        name='embedding_layer'))

        for i in range(parameters.rnn_layers):
            gru = GRU(parameters.rnn_size, return_sequences=True, name='gru_layer_%d' % (i + 1))
            model.add(gru)

        model.add(Lambda(simple_context,
                     mask=lambda inputs, mask: mask[:, parameters.max_len_desc:],
                     output_shape = lambda input_shape: (input_shape[0], parameters.max_len_head, 2*(parameters.rnn_size - parameters.activation_rnn_size)),
                     name='simple_context_layer'))

        vocab_size = word2vec.shape[0]
        model.add(TimeDistributed(Dense(vocab_size, name='time_distributed_layer')))
        
        model.add(Activation('softmax', name='activation_layer'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        K.set_value(model.optimizer.lr, np.float32(parameters.learning_rate))
        print (model.summary())

        return model

def train(model,train_size,val_size,val_step_size,epochs,words_replace_count,model_weights_file_name):

    def OHE_to_indexes(y_val):
        list_of_headline = []
        for each_headline in y_val:
            list_of_word_indexes = np.where(np.array(each_headline)==1)[1]
            list_of_headline.append(list(list_of_word_indexes))
        return list_of_headline

    def indexes_to_words(list_of_headline):

        list_of_word_headline = []
        for each_headline in list_of_headline:
            each_headline_words = []
            for each_word in each_headline:
                if each_word in (empty_tag_location, eos_tag_location, unknown_tag_location):
                    continue
                each_headline_words.append(idx2word[each_word])
            list_of_word_headline.append(each_headline_words)            
        return list_of_word_headline

    def blue_score_text(y_actual,y_predicated):
        assert len(y_actual) ==  len(y_predicated)
        no_of_news = len(y_actual)
        blue_score = 0.0
        for i in range(no_of_news):
            reference = y_actual[i]
            hypothesis = y_predicated[i]

            weights=(0.25, 0.25, 0.25, 0.25)
            min_len_present = min(len(reference),len(hypothesis))
            if min_len_present==0:
                continue
            if min_len_present<4:
                weights=[1.0/min_len_present,]*min_len_present
                
            blue_score = blue_score + sentence_bleu([reference],hypothesis,weights=weights)
            
        return blue_score/float(no_of_news)

    def blue_score_calculator(model, no_of_validation_sample, validation_step_size):
        number_words_to_replace=0
        temp_gen = data_generator(number_words_to_replace, model, type='val')        
            
        total_blue_score = 0.0            
        blue_batches = 0
        blue_number_of_batches = no_of_validation_sample / validation_step_size
        for X_val, y_val in temp_gen:
            y_predicated = model.predict_classes(X_val,batch_size = validation_step_size,verbose = 1)
            y_predicated_words = indexes_to_words(y_predicated)
            list_of_word_headline = indexes_to_words(OHE_to_indexes(y_val))
            assert len(y_val)==len(list_of_word_headline) 

            total_blue_score = total_blue_score + blue_score_text(list_of_word_headline, y_predicated_words)
                
            blue_batches += 1
            if blue_batches >=  blue_number_of_batches:
                break
            if blue_batches%10==0:
                print ("eval for {} out of {}".format(blue_batches, blue_number_of_batches))

        del temp_gen
        return total_blue_score/float(blue_batches)


    if os.path.isfile(model_weights_file_name):
        print ("loading weights already present in {}".format(model_weights_file_name))
        model.load_weights(model_weights_file_name)
        print ("model weights loaded for further training")
            
    train_data = data_generator(words_replace_count, model, type='train')
    blue_scores = []
    best_blue_score_track = -1.0
    number_of_batches = math.ceil(train_size / float(128))
        
    for each_epoch in range(epochs):
        print ("running for epoch ",each_epoch)
        start_time = time.time()

        batches = 0
        for X_batch, Y_batch in train_data:
            model.fit(X_batch,Y_batch,batch_size=128,epochs=1)
            batches += 1
            if batches >= number_of_batches :
                break
            if batches%10==0:
                print ("training for {} out of {} for epoch {}".format(batches, number_of_batches, each_epoch))
                    
        end_time = time.time()
        print("time to train epoch ",end_time-start_time)

        blue_score_now = blue_score_calculator(model,val_size,val_step_size)
        blue_scores.append(blue_score_now)
        if best_blue_score_track < blue_score_now:
            best_blue_score_track = blue_score_now
            print ("saving model for blue score ",best_blue_score_track)
            model.save_weights(model_weights_file_name)
                
        with open("blue_scores.pickle", "wb") as output_file:
            pickle.dump(blue_scores, output_file)

def test(model, no_of_testing_sample, model_weights_file_name,top_k,output_file,seperator='#|#'):

    def indexes_to_words(list_of_headline):
    
        list_of_word_headline = []
        for each_headline in list_of_headline:
            each_headline_words = []
            for each_word in each_headline:
                if each_word in (empty_tag_location, eos_tag_location, unknown_tag_location):
                    continue
                each_headline_words.append(idx2word[each_word])
            list_of_word_headline.append(each_headline_words)            
        return list_of_word_headline

    def is_headline_end(word_index_list,current_predication_position):
    
        if (word_index_list is None) or (len(word_index_list)==0):
            return False
        if word_index_list[current_predication_position]==eos_tag_location or current_predication_position>=parameters.max_length:
            return True
        return False

    def process_word(predication, word_position_index, top_k, X, prev_layer_log_prob):
        
        predication = predication[0]
        predication_at_word_index = predication[word_position_index]
        sorted_arg = predication_at_word_index.argsort()
        top_probable_indexes = sorted_arg[::-1]
        top_probabilities = np.take(predication_at_word_index,top_probable_indexes)
        log_probabilities = np.log(top_probabilities)
        log_probabilities[log_probabilities == -inf] = -sys.maxsize - 1
        log_probabilities = log_probabilities + prev_layer_log_prob
        assert len(log_probabilities)==len(top_probable_indexes)
            
        offset = parameters.max_len_desc + word_position_index+1
        ans = []
        count = 0 
        for i,j in zip(log_probabilities, top_probable_indexes):

            if j in X[parameters.max_len_desc+1:offset][-parameters.dont_repeat_word_in_last:]:
                continue
            if (word_position_index < parameters.min_head_line_gen) and (j in [empty_tag_location, unknown_tag_location, eos_tag_location]):
                continue
            next_input = np.concatenate((X[:offset], [j,]))
            next_input = next_input.reshape((1,next_input.shape[0]))

            if offset!=parameters.max_length:
                next_input = sequence.pad_sequences(next_input, maxlen=parameters.max_length, value=empty_tag_location, padding='post', truncating='post')
            next_input = next_input[0]
            ans.append((i,next_input))
            count = count + 1
            if count>=top_k:
                break
        return ans

    def beam_search(model, X, top_k):

        prev_word_index_top_k = []
        curr_word_index_top_k = []
        done_with_pred = []
        data = X.reshape((1,X.shape[0]))
        predication = model.predict_proba(data,verbose=0)
        prev_word_index_top_k = process_word(predication,0,top_k,X,0.0)

        for i in range(1,parameters.max_len_head):
            for j in range(len(prev_word_index_top_k)):
                probability_now, current_intput = prev_word_index_top_k[j]
                data = current_intput.reshape((1,current_intput.shape[0]))
                predication = model.predict_proba(data,verbose=0)
                next_top_k_for_curr_word = process_word(predication,i,top_k,current_intput,probability_now)
                curr_word_index_top_k = curr_word_index_top_k + next_top_k_for_curr_word
                    
            curr_word_index_top_k = sorted(curr_word_index_top_k,key=itemgetter(0),reverse=True)
            prev_word_index_top_k_temp = curr_word_index_top_k[:top_k]
            curr_word_index_top_k = []
            prev_word_index_top_k = []
            for each_proba, each_word_idx_list in prev_word_index_top_k_temp:
                offset = parameters.max_len_desc+i+1
                if is_headline_end(each_word_idx_list,offset):
                    done_with_pred.append((each_proba, each_word_idx_list))
                else:
                    prev_word_index_top_k.append((each_proba,each_word_idx_list))
                
        done_with_pred = sorted(done_with_pred,key=itemgetter(0),reverse=True)
        done_with_pred = done_with_pred[:top_k]
        return done_with_pred


    model.load_weights(model_weights_file_name)
    print ("model weights loaded")
    test_batch_size = 1
    test_data_generator = data_generator(number_words_to_replace=0, model=None,is_training=False, type='test')
    number_of_batches = math.ceil(no_of_testing_sample / float(test_batch_size))
        
    with codecs.open(output_file, 'w',encoding='utf8') as f:
        batches = 0
        for X_batch, Y_batch in test_data_generator:
            X = X_batch[0]
            Y = Y_batch[0]
            assert X[parameters.max_len_desc]==eos_tag_location
            X[parameters.max_len_desc+1:]=empty_tag_location
            result = beam_search(model,X,top_k)
            list_of_word_indexes = result[0][1]
            list_of_words = indexes_to_words([list_of_word_indexes])[0]
            headline = u" ".join(list_of_words[parameters.max_len_desc+1:])
            f.write(Y+seperator+headline+"\n")
            batches += 1
            if batches >= number_of_batches :
                break
            if batches%10==0:
                print ("working on batch no {} out of {}".format(batches,number_of_batches))



word2vec = load_fasttext_embedding('wiki.fa/wiki.fa.vec')

# model = lstm_model()
model = gru_model()
# model = bidirectional_model()


train(model=model, 
    train_size=parameters.train_size, 
    val_size=parameters.val_size,
    val_step_size=128, 
    epochs=16, 
    words_replace_count=5,
    model_weights_file_name='model_weights.h5')


test(model=model,
    no_of_testing_sample= 12567,
    model_weights_file_name='model_weights.h5',
    top_k=5,
    output_file='test_output.txt')