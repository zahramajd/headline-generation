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
import pandas as pd


from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.wrappers import TimeDistributed
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.core import Lambda, Activation
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.preprocessing import sequence

from tqdm import tqdm
# from sklearn.cross_validation import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from numpy import inf
from operator import itemgetter
import matplotlib.pyplot as plt


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
# initalize end of sentence, empty and unk tokens
word2idx['<empty>'] = empty_tag_location
word2idx['<eos>'] = eos_tag_location
word2idx['<unk>'] = unknown_tag_location
idx2word[empty_tag_location] = '<empty>'
idx2word[eos_tag_location] = '<eos>'
idx2word[unknown_tag_location] = '<unk>'



def read_word_embedding(file_name):

    idx = 3
    temp_word2vec_dict = {}
    # <empty>, <eos> tag replaced by word2vec learning
    # create random dimensional vector for empty, eos and unk tokens
    temp_word2vec_dict['<empty>'] = [float(i) for i in np.random.rand(parameters.embedding_dimension, 1)]
    temp_word2vec_dict['<eos>'] = [float(i) for i in np.random.rand(parameters.embedding_dimension, 1)]
    temp_word2vec_dict['<unk>'] = [float(i) for i in np.random.rand(parameters.embedding_dimension, 1)]
    model = gensim.models.KeyedVectors.load_word2vec_format(file_name, limit = 40000)
    V = model.index2word
    X = np.zeros((parameters.top_freq_word_to_use, model.vector_size))
    for index, word in enumerate(V):
        vector = model[word]
        temp_word2vec_dict[idx] = vector
        word2idx[word] = idx
        idx2word[idx] = word
        idx = idx + 1
        if idx % 10000 == 0:
            print ("working on word2vec ... idx ", idx)
            
    return temp_word2vec_dict

def simple_context(X, mask):
    """
    Simple context calculation layer logic
    X = (batch_size, time_steps, units)
    time_steps are nothing but number of words in our case.
    """
    # segregrate heading and desc
    desc, head = X[:, :parameters.max_len_desc, :], X[:, parameters.max_len_desc:, :]
    # segregrate activation and context part
    head_activations, head_words = head[:, :, :parameters.activation_rnn_size], head[:, :, parameters.activation_rnn_size:]
    desc_activations, desc_words = desc[:, :, :parameters.activation_rnn_size], desc[:, :, parameters.activation_rnn_size:]

    # p=(bacth_size, length_desc_words, rnn_units)
    # q=(bacth_size, length_headline_words, rnn_units)
    # K.dot(p,q) = (bacth_size, length_desc_words,length_headline_words)
    activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2, 2))

    # make sure we dont use description words that are masked out
    activation_energies = activation_energies + -1e20 * K.expand_dims(1. - K.cast(mask[:, :parameters.max_len_desc], 'float32'), 1)

    # for every head word compute weights for every desc word
    activation_energies = K.reshape(activation_energies, (-1, parameters.max_len_desc))
    activation_weights = K.softmax(activation_energies)
    activation_weights = K.reshape(activation_weights, (-1, parameters.max_len_head, parameters.max_len_desc))

    # for every head word compute weighted average of desc words
    desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2, 1))
    return K.concatenate((desc_avg_word, head_words))

def output_shape_simple_context_layer(input_shape):
    return (input_shape[0], parameters.max_len_head , 2 * (parameters.rnn_size - parameters.activation_rnn_size))

def create_model():

        length_vocab, embedding_size = word2vec.shape
        print ("shape of word2vec matrix ", word2vec.shape)

        model = Sequential()
        model.add(
                Embedding(
                        length_vocab, embedding_size,
                        input_length=parameters.max_length,
                        weights=[word2vec], mask_zero=True,
                        name='embedding_layer'
                )
        )

        for i in range(parameters.rnn_layers):
            lstm = LSTM(parameters.rnn_size, return_sequences=True,
                name='lstm_layer_%d' % (i + 1)
            )

            model.add(lstm)
            # No drop out added !

        model.add(Lambda(simple_context,
                     mask=lambda inputs, mask: mask[:, parameters.max_len_desc:],
                     output_shape=output_shape_simple_context_layer,
                     name='simple_context_layer'))

        vocab_size = word2vec.shape[0]
        model.add(TimeDistributed(Dense(vocab_size,
                                name='time_distributed_layer')))
        
        model.add(Activation('softmax', name='activation_layer'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        K.set_value(model.optimizer.lr, np.float32(parameters.learning_rate))
        print (model.summary())

        return model

def padding(list_idx, curr_max_length, is_left):
    """
    padds with <empty> tag in left side
    """
    if len(list_idx) >= curr_max_length:
        return list_idx
    number_of_empty_fill = curr_max_length - len(list_idx)
    if is_left:
        return [empty_tag_location, ] * number_of_empty_fill + list_idx
    else:
        return list_idx + [empty_tag_location, ] * number_of_empty_fill

def headline2idx(list_idx, curr_max_length, is_input):
    """
    if space add <eos> tag in input case, input size = curr_max_length-1
    always add <eos> tag in predication case, size = curr_max_length
    always right pad
    """
    if is_input:
        if len(list_idx) >= curr_max_length - 1:
            return list_idx[:curr_max_length - 1]
        else:
            # space remaning add eos and empty tags
            list_idx = list_idx + [eos_tag_location, ]
            return padding(list_idx, curr_max_length - 1, False)
    else:
        # always add <eos>
        if len(list_idx) == curr_max_length:
            list_idx[-1] = eos_tag_location
            return list_idx
        else:
            # space remaning add eos and empty tags
            list_idx = list_idx + [eos_tag_location, ]
            return padding(list_idx, curr_max_length, False)

def desc2idx(list_idx, curr_max_length):
    """
    always left pad and eos tag to end
    """
    #====== REVERSE THE DESC IDS ========
    list_idx.reverse()
    # padding to the left remain same and 
    # eos tag position also remain same,
    # just description flipped
    #===================================
    # desc padded left
    list_idx = padding(list_idx, curr_max_length, True)
    # eos tag add
    list_idx = list_idx + [eos_tag_location, ]
    return list_idx

def sentence2idx(sentence, is_headline, curr_max_length, is_input=True):
    """
    given a sentence convert it to its ids
    "I like India" => [12, 51, 102]
    if words not present in vocab ignore them
    is_input is only for headlines
    """
    list_idx = []
    tokens = sentence.split(" ")
    count = 0
    for each_token in tokens:
        if each_token in word2idx:
            list_idx.append(word2idx[each_token])
        else:
            #append unk token as original word not present in word2vec
            list_idx.append(word2idx['<unk>'])
        count = count + 1
        if count >= curr_max_length:
            break

    if is_headline:
        return headline2idx(list_idx, curr_max_length, is_input)
    else:
        return desc2idx(list_idx, curr_max_length)

def flip_words_randomly(description_headline_data, number_words_to_replace, model):

    if number_words_to_replace <= 0 or model == None:
        return description_headline_data

    # check all descrption ends with <eos> tag else throw error
    assert np.all(description_headline_data[:, parameters.max_len_desc] == eos_tag_location)

    batch_size = len(description_headline_data)
    predicated_headline_word_idx = model.predict(description_headline_data, verbose=1, batch_size = batch_size)
    copy_data = description_headline_data.copy()
    for idx in range(batch_size):
        # description = 0 ... max_len_desc-1
        # <eos> = max_len_desc
        # headline = max_len_desc + 1 ...
        random_flip_pos = sorted(random.sample(range(parameters.max_len_desc + 1, parameters.max_length), number_words_to_replace))
        for replace_idx in random_flip_pos:
            # Don't replace <eos> and <empty> tag
            if (description_headline_data[idx, replace_idx] == empty_tag_location or
            description_headline_data[idx, replace_idx] == eos_tag_location):
                continue

            # replace_idx offset moving as predication doesnot have desc
            new_id = replace_idx - (parameters.max_len_desc + 1)
            prob_words = predicated_headline_word_idx[idx, new_id]
            word_idx = prob_words.argmax()
            # dont replace by empty location or eos tag location
            if word_idx == empty_tag_location or word_idx == eos_tag_location:
                continue
            copy_data[idx, replace_idx] = word_idx
    return copy_data

def convert_inputs(descriptions, headlines,number_words_to_replace, model,is_training):

    # length of headlines and descriptions should be equal
    assert len(descriptions) == len(headlines)

    X, y = [], []
    for each_desc, each_headline in zip(descriptions, headlines):
        # print('each ', type(each_headline[0]))

        input_headline_idx = sentence2idx(each_headline[0], True, parameters.max_len_head, True)
        predicted_headline_idx = sentence2idx(each_headline[0], True, parameters.max_len_head, False)
        desc_idx = sentence2idx(each_desc[0], False, parameters.max_len_desc)

        # assert size checks
        assert len(input_headline_idx) == parameters.max_len_head - 1
        assert len(predicted_headline_idx) == parameters.max_len_head
        assert len(desc_idx) == parameters.max_len_desc + 1

        X.append(desc_idx + input_headline_idx)
        y.append(predicted_headline_idx)
        
    X, y = np.array(X), np.array(y)
    if is_training:
        #print("Length of X before flipping",len(X))
        X = flip_words_randomly(X, number_words_to_replace, model)
        # One hot encoding of y
        vocab_size = word2vec.shape[0]
        length_of_data = len(headlines)
        Y = np.zeros((length_of_data, parameters.max_len_head, vocab_size))
        for i, each_y in enumerate(y):
            Y[i, :, :] = np_utils.to_categorical(each_y, vocab_size)
        #check equal lengths
        assert len(X)==len(Y)
        return X, Y
    else:
        #Testing doesnot require OHE form of headline, flipping also not required
        #Because BLUE score require words and not OHE form to check accuracy
        return X,headlines

def large_file_reading_generator(data):
    """
    read large file line by line
    """
    while True:
        for each_line in data.items():
            yield each_line
        #shuffle_file(file_name)

# def data_generator(file_name,number_words_to_replace,model,is_training=True):

#     with open("HEADLINE_Generation/sample.headline.data", "r") as fp:
#         headlines = fp.readlines()
#         # headlines = headlines[0:100]
#         fp.close()

#     with open("HEADLINE_Generation/sample.content.data", "r") as fp:
#         contents = fp.readlines()
#         # contents = contents[0:100]
#         fp.close()

#     while len(headlines) > 0:
#         X, Y = [], []
#         for i in range(128):
#             if(len(headlines)<1): continue
#             else:
#                 heads_line = headlines.pop()
#                 descs_line = contents.pop()
#                 X.append(descs_line)
#                 Y.append(heads_line)
#         yield convert_inputs(X, Y, number_words_to_replace, model,is_training)


def data_generator(file_name,number_words_to_replace,model,is_training=True):

    # with open(file_name,'rb') as file_pointer:
    #     data = pickle.load(file_pointer)
    #     headlines_data = data['heads']
    #     descs_data = data['descs']

    data = pd.read_csv(file_name, header=None)[0:100]
    headlines_data = data.iloc[:, [1]]
    descs_data = data.iloc[:, [0]]

    headline_iterator = large_file_reading_generator(headlines_data)
    descs_iterator = large_file_reading_generator(descs_data)
    while True:
        X, y = [], []
        for i in range(128):
            heads_line = next(headline_iterator)
            descs_line = next(descs_iterator)
            heads_line = heads_line[1]
            descs_line = descs_line[1]
            X.append(descs_line)
            y.append(heads_line)
        #print(y)
        yield convert_inputs(X, y, number_words_to_replace, model,is_training)

def OHE_to_indexes(y_val):
    """
    reverse of OHE 
    OHE => indexes
    e.g. [[0,0,1],[1,0,0]] => [2,0]
    """
    list_of_headline = []
    for each_headline in y_val:
        list_of_word_indexes = np.where(np.array(each_headline)==1)[1]
        list_of_headline.append(list(list_of_word_indexes))
    return list_of_headline

def indexes_to_words(list_of_headline):
    """
    indexes => words (for BLUE Score)
    e.g. [2,0] => ["I","am"] (idx2word defined dictionary used)
    """
    list_of_word_headline = []
    for each_headline in list_of_headline:
        each_headline_words = []
        for each_word in each_headline:
            #Dont include <eos> and <empty> tags
            if each_word in (empty_tag_location, eos_tag_location, unknown_tag_location):
                continue
            each_headline_words.append(idx2word[each_word])
        list_of_word_headline.append(each_headline_words)            
    return list_of_word_headline

def blue_score_text(y_actual,y_predicated):
    #check length equal
    assert len(y_actual) ==  len(y_predicated)
    #list of healine .. each headline has words
    no_of_news = len(y_actual)
    blue_score = 0.0
    for i in range(no_of_news):
        reference = y_actual[i]
        hypothesis = y_predicated[i]
            
        #Avoid ZeroDivisionError in blue score
        #default weights
        weights=(0.25, 0.25, 0.25, 0.25)
        min_len_present = min(len(reference),len(hypothesis))
        if min_len_present==0:
            continue
        if min_len_present<4:
            weights=[1.0/min_len_present,]*min_len_present
            
        blue_score = blue_score + sentence_bleu([reference],hypothesis,weights=weights)
        
    return blue_score/float(no_of_news)

def blue_score_calculator(model, validation_file_name, no_of_validation_sample, validation_step_size):
    #In validation don't repalce with random words
    number_words_to_replace=0
    temp_gen = data_generator(validation_file_name,number_words_to_replace, model)        
        
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
            #get out of infinite loop of val generator
            break
        if blue_batches%10==0:
            print ("eval for {} out of {}".format(blue_batches, blue_number_of_batches))

    #close files and delete generator  
    del temp_gen
    return total_blue_score/float(blue_batches)

def train(model,data_file,val_file,train_size,val_size,val_step_size,epochs,words_replace_count,model_weights_file_name):

    #load model weights if file present 
    if os.path.isfile(model_weights_file_name):
        print ("loading weights already present in {}".format(model_weights_file_name))
        model.load_weights(model_weights_file_name)
        print ("model weights loaded for further training")
            
    train_data = data_generator(data_file,words_replace_count, model)
    #print(train_data)
    blue_scores = []
    #blue score are always greater than 0
    best_blue_score_track = -1.0
    number_of_batches = math.ceil(train_size / float(128))
        
    for each_epoch in range(epochs):
        print ("running for epoch ",each_epoch)
        start_time = time.time()
        #print(start_time)
        #manually loop over batches and feed to network
        #purposefully not used fit_generator
        batches = 0
        for X_batch, Y_batch in train_data:

            model.fit(X_batch,Y_batch,batch_size=128,epochs=1)
            batches += 1
            #take last chunk and roll over to start ...
            #therefore float used ... 
            if batches >= number_of_batches :
                break
            if batches%10==0:
                print ("training for {} out of {} for epoch {}".format(batches, number_of_batches, each_epoch))
                    
        end_time = time.time()
        print("time to train epoch ",end_time-start_time)

        # evaluate model on BLUE score and save best BLUE score model...
        blue_score_now = blue_score_calculator(model,val_file,val_size,val_step_size)
        blue_scores.append(blue_score_now)
        if best_blue_score_track < blue_score_now:
            best_blue_score_track = blue_score_now
            print ("saving model for blue score ",best_blue_score_track)
            model.save_weights(model_weights_file_name)
                
        # Note : It saves on every loop, this looks REPETATIVE, BUT
        # if user aborts(control-c) in middle of epochs then we get previous
        # present history
        # User can track previous history while model running ... 
        # dump history object list for further plotting of loss
        # append BLUE Score for to another list  and dump for futher plotting
        with open("blue_scores.pickle", "wb") as output_file:
            pickle.dump(blue_scores, output_file)

    return blue_scores

def is_headline_end(word_index_list,current_predication_position):
    """
    is headline ended checker
    current_predication_position is 0 index based
    """
    if (word_index_list is None) or (len(word_index_list)==0):
        return False
    if word_index_list[current_predication_position]==eos_tag_location or current_predication_position>=parameters.max_length:
        return True
    return False

def process_word(predication,word_position_index,top_k,X,prev_layer_log_prob):
    """
    Extract top k predications of given position
    """
    #predication conttains only one element
    #shape of predication (1,max_head_line_words,vocab_size)
    predication = predication[0]
    #predication (max_head_line_words,vocab_size)
    predication_at_word_index = predication[word_position_index]
    #http://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
    sorted_arg = predication_at_word_index.argsort()
    top_probable_indexes = sorted_arg[::-1]
    top_probabilities = np.take(predication_at_word_index,top_probable_indexes)
    log_probabilities = np.log(top_probabilities)
    #make sure elements doesnot contain -infinity
    log_probabilities[log_probabilities == -inf] = -sys.maxsize - 1
    #add prev layer probability
    log_probabilities = log_probabilities + prev_layer_log_prob
    assert len(log_probabilities)==len(top_probable_indexes)
        
    #add previous words ... preparation for next input
    #offset calculate ... description + eos + headline till now
    offset = parameters.max_len_desc + word_position_index+1
    ans = []
    count = 0 
    for i,j in zip(log_probabilities, top_probable_indexes):
        #check for word should not repeat in headline ... 
        #checking for last x words, where x = parameters.dont_repeat_word_in_last
        if j in X[parameters.max_len_desc+1:offset][-parameters.dont_repeat_word_in_last:]:
            continue
        if (word_position_index < parameters.min_head_line_gen) and (j in [empty_tag_location, unknown_tag_location, eos_tag_location]):
            continue
        next_input = np.concatenate((X[:offset], [j,]))
        next_input = next_input.reshape((1,next_input.shape[0]))
        #for the last time last word put at max_length + 1 position 
        #don't truncate that
        if offset!=parameters.max_length:
            next_input = sequence.pad_sequences(next_input, maxlen=parameters.max_length, value=empty_tag_location, padding='post', truncating='post')
        next_input = next_input[0]
        ans.append((i,next_input))
        count = count + 1
        if count>=top_k:
            break
    #[(prob,list_of_words_as_next_input),(prob2,list_of_words_as_next_input2),...]
    return ans

def beam_search(model,X,top_k):
    """
    1.Loop over max headline word allowed
    2.predict word prob and select top k words for each position
    3.select top probable combination uptil now for next round
    """
    #contains [(log_p untill now, word_seq), (log_p2, word_seq2)]
    prev_word_index_top_k = []
    curr_word_index_top_k = []
    done_with_pred = []
    #1d => 2d array [1,2,3] => [[1,2,3]]
    data = X.reshape((1,X.shape[0]))
    #shape of predication (1,max_head_line_words,vocab_size)
    predication = model.predict_proba(data,verbose=0)
    #prev layer probability 1 => np.log(0)=0.0
    prev_word_index_top_k = process_word(predication,0,top_k,X,0.0)

    #1st time its done above to fill prev word therefore started from 1
    for i in range(1,parameters.max_len_head):
        #i = represents current intrested layer ...
        for j in range(len(prev_word_index_top_k)):
            #j = each time loops for top k results ...
            probability_now, current_intput = prev_word_index_top_k[j]
            data = current_intput.reshape((1,current_intput.shape[0]))
            predication = model.predict_proba(data,verbose=0)
            next_top_k_for_curr_word = process_word(predication,i,top_k,current_intput,probability_now)
            curr_word_index_top_k = curr_word_index_top_k + next_top_k_for_curr_word
                
        #sort new list, empty old, copy top k element to old, empty new
        curr_word_index_top_k = sorted(curr_word_index_top_k,key=itemgetter(0),reverse=True)
        prev_word_index_top_k_temp = curr_word_index_top_k[:top_k]
        curr_word_index_top_k = []
        prev_word_index_top_k = []
        #if word predication eos ... put it done list ...
        for each_proba, each_word_idx_list in prev_word_index_top_k_temp:
            offset = parameters.max_len_desc+i+1
            if is_headline_end(each_word_idx_list,offset):
                done_with_pred.append((each_proba, each_word_idx_list))
            else:
                prev_word_index_top_k.append((each_proba,each_word_idx_list))
            
    #sort according to most probable
    done_with_pred = sorted(done_with_pred,key=itemgetter(0),reverse=True)
    done_with_pred = done_with_pred[:top_k]
    return done_with_pred

def test(model, data_file_name, no_of_testing_sample, model_weights_file_name,top_k,output_file,seperator='#|#'):
    """
    test on given description data file with empty headline ...
    """
    model.load_weights(model_weights_file_name)
    print ("model weights loaded")
    #Always 1 for now ... later batch code for test sample created
    test_batch_size = 1
    test_data_generator = data_generator(data_file_name, number_words_to_replace=0, model=None,is_training=False)
    number_of_batches = math.ceil(no_of_testing_sample / float(test_batch_size))
        
    with codecs.open(output_file, 'w',encoding='utf8') as f:
        #testing batches
        batches = 0
        for X_batch, Y_batch in test_data_generator:
            #Always come one because X_batch contains one element
            X = X_batch[0]
            Y = Y_batch[0]
            assert X[parameters.max_len_desc]==eos_tag_location
            #wipe up news headlines present and replace by empty tag ...            
            X[parameters.max_len_desc+1:]=empty_tag_location
            result = beam_search(model,X,top_k)
            #take top most probable element
            list_of_word_indexes = result[0][1]
            list_of_words = indexes_to_words([list_of_word_indexes])[0]
            headline = u" ".join(list_of_words[parameters.max_len_desc+1:])
            f.write(Y+seperator+headline+"\n")
            batches += 1
            #take last chunk and roll over to start ...
            #therefore float used ... 
            if batches >= number_of_batches :
                break
            if batches%10==0:
                print ("working on batch no {} out of {}".format(batches,number_of_batches))




temp_word2vec_dict = read_word_embedding('wiki.fa/wiki.fa.vec')
length_vocab = len(temp_word2vec_dict)
shape = (length_vocab, parameters.embedding_dimension)
# faster initlization and random for <empty> and <eos> tag
word2vec = np.random.uniform(low=-1, high=1, size=shape)
for i in range(length_vocab):
    if i in temp_word2vec_dict:
        word2vec[i, :] = temp_word2vec_dict[i]

model = create_model()

blue_scores =train(model=model, 
    data_file='train.csv', 
    val_file='val.csv',
    train_size=8910, 
    val_size=990,
    val_step_size=128, 
    epochs=16, 
    words_replace_count=5,
    model_weights_file_name='lstm_weights.h5')



test(model=model,
    data_file_name='test.csv',
    no_of_testing_sample= 100,
    model_weights_file_name='lstm_weights.h5',
    top_k=5,
    output_file='test_output_lstm.txt')


# plot 
N = len(blue_scores)
plt.style.use("ggplot")
fig = plt.figure(figsize=(20,8))
plt.title("BLUE, LSTM")
plt.plot(np.arange(0, N), blue_scores)

plt.xlabel("Epoch #")
plt.ylabel("BLUE score")
plt.legend(loc="lower left")
plt.show()