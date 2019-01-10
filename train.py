import pandas as pd
import numpy as np
import string
from string import digits
import re
from sklearn.cross_validation import train_test_split
import pickle

from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.utils import plot_model

import keras
print(keras.__version__)
import argparse

#parser = argparse.ArgumentParser(description='Process some integers.')
#parser.add_argument('--input_file', xrequired=True, help='Input data file, should be in the format <input_string>\\t<output_string>')
#args = parser.parse_args()


MINI_BATCH_SIZE=100
MAX_LENGTH=72
EMBEDDING_SIZE=72



def get_token_index(filename):
    lines= pd.read_table(filename, names=['inp', 'outp'])
    lines.inp=lines.inp.apply(lambda x: x.lower())
    lines.outp=lines.outp.apply(lambda x: x.lower())
    lines.inp=lines.inp.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
    lines.outp=lines.outp.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
    exclude = set(string.punctuation)
    lines.inp=lines.inp.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    lines.outp=lines.outp.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    lines.outp = lines.outp.apply(lambda x : 'START_ '+ x + ' _END')
    all_words=set()
    for inp in lines.inp:
        for word in inp.split():
            if word not in all_words:
                all_words.add(word)
        
    for outp in lines.outp:
        for word in outp.split():
            if word not in all_words:
                all_words.add(word)
    token_index = dict([(word, i) for i, word in enumerate(all_words)])
    return lines, token_index, all_words


def build_train_model():
    encoder_inputs = Input(shape=(None,))
    en_x=  Embedding(num_tokens, EMBEDDING_SIZE)(encoder_inputs)
    encoder = LSTM(MAX_LENGTH, return_state=True)
    encoder_outputs, state_h, state_c = encoder(en_x)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))

    dex=  Embedding(num_tokens, EMBEDDING_SIZE)

    final_dex= dex(decoder_inputs)


    decoder_lstm = LSTM(MAX_LENGTH, return_sequences=True, return_state=True)

    decoder_outputs, _, _ = decoder_lstm(final_dex,
                                        initial_state=encoder_states)

    decoder_dense = Dense(num_tokens, activation='softmax')

    decoder_outputs = decoder_dense(decoder_outputs)

    train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    return train_model, encoder_inputs, encoder_states, decoder_inputs, dex, decoder_lstm, decoder_dense


def train_mini_batch(lines, num_tokens, train_model):
    encoder_input_data = np.zeros(
    (len(lines.inp), MAX_LENGTH),
    dtype='float32')
    decoder_input_data = np.zeros(
    (len(lines.outp), MAX_LENGTH),
    dtype='float32')
    decoder_target_data = np.zeros(
    (len(lines.outp), MAX_LENGTH, num_tokens),
    dtype='float32')
    print(len(lines))
    for i, (input_text, target_text) in enumerate(zip(lines.inp, lines.outp)):
        for t, word in enumerate(input_text.split()):
            encoder_input_data[i, t] = token_index[word]
    for t, word in enumerate(target_text.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = token_index[word]

        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, token_index[word]] = 1.

    train_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=128,
          epochs=2,
          validation_split=0.05)





lines, token_index, all_words = get_token_index('output1.txt')
print(len(lines))
words = sorted(list(all_words))
num_tokens = len(all_words)

print(len(lines.outp)*MAX_LENGTH*num_tokens)

train_model, encoder_inputs, encoder_states, decoder_inputs, dex, decoder_lstm, decoder_dense= build_train_model()

train_model.summary()


train_mini_batch(lines[:MINI_BATCH_SIZE], num_tokens, train_model)
train_mini_batch(lines[MINI_BATCH_SIZE:MINI_BATCH_SIZE+MINI_BATCH_SIZE], num_tokens, train_model)

print('encoder_inputs={}, encoder_states={}'.format(type(encoder_inputs),type(encoder_states)))
encoder_model = Model(encoder_inputs, encoder_states)
print(encoder_model.summary())


decoder_state_input_h = Input(shape=(MAX_LENGTH,))
decoder_state_input_c = Input(shape=(MAX_LENGTH,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

final_dex2= dex(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)
print(decoder_model.summary())


with open('encoder_model.json', 'w', encoding='utf8') as f:
    f.write(encoder_model.to_json())
encoder_model.save_weights('encoder_model_weights.h5')

with open('decoder_model.json', 'w', encoding='utf8') as f:
    f.write(decoder_model.to_json())
decoder_model.save_weights('decoder_model_weights.h5')

with open('token_index.p', 'wb') as fp:
    pickle.dump(token_index, fp, protocol=pickle.HIGHEST_PROTOCOL)