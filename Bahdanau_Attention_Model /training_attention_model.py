# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import string

import matplotlib.pyplot as plt
# %matplotlib inline

import tensorflow as tf

from sklearn.model_selection import train_test_split
import re
import os

from google.colab import drive
drive.mount('/content/drive')

"""#### Import cleaned data"""

df = pd.read_csv("/content/drive/MyDrive/ML/rnn/Machine_Translation/Data/cleaned.csv")

df.tail()

"""**Note-** this dataset was cleaned in notebook1 [here](https://github.com/AdiShirsath/Neural-Machine-Translation/blob/master/EDA_And_Cleaning_Text.ipynb)"""

df.info()

"""### Add start and end tokens to target sentecnes
* This helps model understand when sentence is started and ended
* Beause of this decoder can handle diff length sentence than encoder
"""

df['Marathi'] =df.Marathi.apply(lambda x: 'sos '+ x + ' eos')

df.head()

"""##### Convert to list for tokenizer"""

eng_texts = df.English.to_list()
mar_texts = df.Marathi.to_list()

"""## Tokenizer
* Converting into numbers
"""

from tensorflow.keras.preprocessing.text import Tokenizer

def tokenize_sent(text):
  '''
  Take list on texts as input and
  returns its tokenizer and enocded text
  '''
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(text)

  return tokenizer, tokenizer.texts_to_sequences(text)

# Tokenize english and marathi sentences
eng_tokenizer, eng_encoded= tokenize_sent(text= eng_texts)
mar_tokenizer, mar_encoded= tokenize_sent(text= mar_texts)

"""#### English"""

eng_encoded[100:105]

eng_index_word = eng_tokenizer.index_word
eng_word_indec= eng_tokenizer.word_index

"""### Get vocab size which will be needed next"""

ENG_VOCAB_SIZE = len(eng_tokenizer.word_counts)+1
ENG_VOCAB_SIZE

"""#### Mrathi"""

mar_encoded[30000:30005]

mar_index_word = mar_tokenizer.index_word
mar_word_index= mar_tokenizer.word_index

MAR_VOCAB_SIZE=len(mar_tokenizer.word_counts)+1
MAR_VOCAB_SIZE

max_eng_len = 0
for i in range(len(eng_encoded)):
  if len(eng_encoded[i]) > max_eng_len:
    max_eng_len= len(eng_encoded[i])

max_mar_len = 0
for i in range(len(mar_encoded)):
  if len(eng_encoded[i]) > max_mar_len:
    max_mar_len= len(mar_encoded[i])

print(max_eng_len)
max_mar_len

"""## Padding
* Making input sentences as max length of input sentence with padding zero
* Same for target make them as max length of target sentence.
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences

eng_padded = pad_sequences(eng_encoded, maxlen=max_eng_len, padding='post')
mar_padded = pad_sequences(mar_encoded, maxlen=max_mar_len, padding='post')

eng_padded

mar_padded.shape

"""##### Converting to array"""

eng_padded= np.array(eng_padded)
mar_padded= np.array(mar_padded)

"""## Train test split"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(eng_padded, mar_padded, test_size=0.1, random_state=0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

"""## Building Model"""

from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Concatenate, Dropout
from tensorflow.keras import Input, Model

### Import attention layer

os.chdir("/content/drive/MyDrive/ML/rnn/Machine_Translation/Attention")
from BahdanauAttention import AttentionLayer

# Encoder

encoder_inputs = Input(shape=(max_eng_len,))
enc_emb = Embedding(ENG_VOCAB_SIZE, 1024)(encoder_inputs)

# Bidirectional lstm layer
enc_lstm1 = Bidirectional(LSTM(256,return_sequences=True,return_state=True))
encoder_outputs1, forw_state_h, forw_state_c, back_state_h, back_state_c = enc_lstm1(enc_emb)

final_enc_h = Concatenate()([forw_state_h,back_state_h])
final_enc_c = Concatenate()([forw_state_c,back_state_c])

encoder_states =[final_enc_h, final_enc_c]

# Set up the decoder.
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(MAR_VOCAB_SIZE, 1024)
dec_emb = dec_emb_layer(decoder_inputs)
#LSTM using encoder_states as initial state
decoder_lstm = LSTM(512, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

#Attention Layer
attention_layer = AttentionLayer()
attention_result, attention_weights = attention_layer([encoder_outputs1, decoder_outputs])

# Concat attention output and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_result])

#Dense layer
decoder_dense = Dense(MAR_VOCAB_SIZE, activation='softmax')
decoder_outputs = decoder_dense(decoder_concat_input)


# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model,show_shapes=True)

"""***IMP note :- if loss categorical crossentropy used then shapes incompatible error will occcur beause we have to use sparse_categorical_crossentropy when we have all different labels categorical is for mutliclass labels***"""

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

"""### Define callbacks"""

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("/content/drive/MyDrive/ML/rnn/Machine_Translation/Attention/model_checkpoints/model1/", monitor='val_accuracy')

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

callbacks_list = [checkpoint, early_stopping]

# Training
encoder_input_data = X_train
# To make same as target data skip last number which is just padding
decoder_input_data = y_train[:,:-1]
# Decoder target data has to be one step ahead so we are taking from 1 as told in keras docs
decoder_target_data =  y_train[:,1:]

# Testing
encoder_input_test = X_test
decoder_input_test = y_test[:,:-1]
decoder_target_test=  y_test[:,1:]

EPOCHS= 50 #@param {type:'slider',min:10,max:100, step:10 }

history = model.fit([encoder_input_data, decoder_input_data],decoder_target_data,
                    epochs=EPOCHS,
                    batch_size=128,
                    validation_data = ([encoder_input_test, decoder_input_test],decoder_target_test),
                    callbacks= callbacks_list)

"""#### Saving weights is very important if ypu dont after colab session ends you might have to retrain model"""

model.save_weights("/content/drive/MyDrive/ML/rnn/Machine_Translation/Attention/saved_model/model1.h5")

"""#### After saving weight you can restart colab session without GPU
Create model and
load model
"""

model.load_weights("/content/drive/MyDrive/ML/rnn/Machine_Translation/Attention/saved_model/model1.h5")

"""## Inference model
* For prediction we have to do this was because we trained encoder on input and decoder on target differently so we'll have to do same for prediction
"""

encoder_model = Model(encoder_inputs, outputs = [encoder_outputs1, final_enc_h, final_enc_c])

decoder_state_h = Input(shape=(512,))
decoder_state_c = Input(shape=(512,))
decoder_hidden_state_input = Input(shape=(36,512))

dec_states = [decoder_state_h, decoder_state_c]

dec_emb2 = dec_emb_layer(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=dec_states)

# Attention inference
attention_result_inf, attention_weights_inf = attention_layer([decoder_hidden_state_input, decoder_outputs2])

decoder_concat_input_inf = Concatenate(axis=-1, name='concat_layer')([decoder_outputs2, attention_result_inf])

dec_states2= [state_h2, state_c2]

decoder_outputs2 = decoder_dense(decoder_concat_input_inf)

decoder_model= Model(
                    [decoder_inputs] + [decoder_hidden_state_input, decoder_state_h, decoder_state_c],
                     [decoder_outputs2]+ dec_states2)

"""### Model will predict numbers and word at time so we'll have to convert them to words of language"""

def get_predicted_sentence(input_seq):
    # Encode the input as state vectors.
    enc_output, enc_h, enc_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = mar_word_index['sos']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [enc_output, enc_h, enc_c ])
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
          break
        else:
            # convert max index number to marathi word
            sampled_char = mar_index_word[sampled_token_index]

        if (sampled_char!='end'):
            # aapend it ti decoded sent
            decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length or find stop token.
        if (sampled_char == 'eos' or len(decoded_sentence.split()) >= 36):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        enc_h, enc_c = h, c

    return decoded_sentence

"""## Converting x and y back to words-sentences"""

def get_marathi_sentence(input_sequence):
    sentence =''
    for i in input_sequence:
      if i!=0 :
        sentence =sentence +mar_index_word[i]+' '
    return sentence

def get_english_sentence(input_sequence):
    sentence =''
    for i in input_sequence:
      if i!=0:
        sentence =sentence +eng_index_word[i]+' '
    return sentence

"""# Model results"""

len(X_test)

"""#### Using simple loop we will take random 15 numbers from x_test and get results

"""

for i in np.random.randint(10, 1000, size=15):
  print("English Sentence:",get_english_sentence(X_test[i]))
  print("Actual Marathi Sentence:",get_marathi_sentence(y_test[i])[4:-4])
  # Before passing input it has to be reshape as following
  print("Predicted Marathi Translation:",get_predicted_sentence(X_test[i].reshape(1,36))[:-4])
  print("----------------------------------------------------------------------------------------")