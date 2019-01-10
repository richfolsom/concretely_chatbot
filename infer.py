import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import re
import pickle

from keras.models import model_from_json

def open_files(encoder_model_file, decoder_model_file, token_index_file):
    
    with open(encoder_model_file + '.json', 'r', encoding='utf8') as f:
        encoder_model = model_from_json(f.read())
    encoder_model.load_weights(encoder_model_file+'_weights.h5')
    with open(decoder_model_file + '.json', 'r', encoding='utf8') as f:
        decoder_model = model_from_json(f.read())
    decoder_model.load_weights(decoder_model_file+'_weights.h5')


    with open('token_index.p', 'rb') as pickle_file:
        token_index = pickle.load(pickle_file)

    return encoder_model, decoder_model, token_index

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_token_index[sampled_token_index]
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 180):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence



encoder_model, decoder_model, token_index= open_files('encoder_model', 'decoder_model', 'token_index.p')


reverse_token_index = dict(
    (i, word) for word, i in token_index.items())


print(encoder_model.summary())
print(decoder_model.summary())

input = np.zeros(72)
input_text = 'atamazonhelp where is my order i have been waiting for weeks to receive it and it still has not arrived'
for i, word in enumerate(input_text.split()):
        input[i] = token_index[word]

print(input_text)
print(decode_sequence(input))