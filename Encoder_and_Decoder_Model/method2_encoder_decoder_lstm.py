

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# get data
df= pd.read_csv("/content/drive/MyDrive/ML/rnn/Machine_Translation/Data/cleaned.csv")

df.tail()


df.Marathi = df.Marathi.apply(lambda x: 'sos '+ x +' eos')

# get english and marathi in one list
eng_texts = df.English.to_list()
mar_texts = df.Marathi.to_list()

"""## Tokenizer
Deep learning NN's does not accept text so first have to convert them into numbers
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

eng_tokenizer, eng_encoded= tokenize_sent(text= eng_texts)
mar_tokenizer, mar_encoded= tokenize_sent(text= mar_texts)

eng_encoded[100:105]

eng_index_word = eng_tokenizer.index_word

ENG_VOCAB_SIZE = len(eng_tokenizer.word_counts)+1
ENG_VOCAB_SIZE

mar_encoded[30000:30005]

mar_index_word= mar_tokenizer.index_word

mar_word_index =mar_tokenizer.word_index

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
We can not sent sentences with different length in neural net so padd them with zero
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Use max length for padding for eng and marathi
eng_padded = pad_sequences(eng_encoded, maxlen=max_eng_len, padding='post')
mar_padded = pad_sequences(mar_encoded, maxlen=max_mar_len, padding='post')

eng_padded.shape

mar_padded.shape

# Convert them into array
eng_padded= np.array(eng_padded)
mar_padded= np.array(mar_padded)

"""### Split into train and test"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(eng_padded, mar_padded, test_size=0.1, random_state=0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

"""## Model"""

from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding, Bidirectional, Add, Concatenate, Dropout
from tensorflow.keras import Input, Model

# Eoncoder
encoder_input = Input(shape=(None, ))
encoder_embd = Embedding(ENG_VOCAB_SIZE,512, mask_zero=True)(encoder_input)
encoder_lstm = Bidirectional(LSTM(256, return_state=True))
encoder_output, forw_state_h, forw_state_c, back_state_h, back_state_c = encoder_lstm(encoder_embd)
state_h_final = Concatenate()([forw_state_h, back_state_h])
state_c_final = Concatenate()([forw_state_c, back_state_c])

## Now take only states and create context vector
encoder_states= [state_h_final, state_c_final]

# Decoder
decoder_input = Input(shape=(None,))
# For zero padding we have added +1 in marathi vocab size
decoder_embd = Embedding(MAR_VOCAB_SIZE, 512, mask_zero=True)
decoder_embedding= decoder_embd(decoder_input)
# We used bidirectional layer above so we have to double units of this lstm
decoder_lstm = LSTM(512, return_state=True,return_sequences=True )
# just take output of this decoder dont need self states
decoder_outputs, _, _= decoder_lstm(decoder_embedding, initial_state=encoder_states)
# here this is going to predicct so we can add dense layer here
# here we want to convert predicted numbers into probability so use softmax
decoder_dense= Dense(MAR_VOCAB_SIZE, activation='softmax')
# We will again feed predicted output into decoder to predict its next word
decoder_outputs = decoder_dense(decoder_outputs)

model5 = Model([encoder_input, decoder_input], decoder_outputs)

from tensorflow.keras.utils import plot_model
plot_model(model5)

model5.summary()

y_train

model5.summary()

model5.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

"""#### Callbacks"""

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("/content/drive/MyDrive/ML/rnn/Machine_Translation/Encoder_decoder/model_checkpoints/model5/", monitor='val_accuracy')

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

callbacks_list = [checkpoint, early_stopping]

EPOCHS= 50 #@param {type:'slider',min:10,max:100, step:10 }

"""### Prepare input for encoder -decoder model"""

# Training
encoder_input_data = X_train
decoder_input_data = y_train[:,:-1]
decoder_target_data = y_train[:,1:]

# Testing
encoder_input_test = X_test
decoder_input_test = y_test[:,:-1]
decoder_target_test= y_test[:,1:]

history = model5.fit([encoder_input_data, decoder_input_data],decoder_target_data,
                    epochs=EPOCHS,
                    batch_size=128,
                    validation_data = ([encoder_input_test, decoder_input_test],decoder_target_test ),
                     callbacks= callbacks_list)

model5.save_weights("/content/drive/MyDrive/ML/rnn/Machine_Translation/Encoder_decoder/saved_models/model5")

"""# inference Model"""

from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding
from tensorflow.keras import Input, Model

encoder_model = Model(encoder_input, encoder_states)

decoder_state_input_h = Input(shape=(512,))
decoder_state_input_c= Input(shape=(512,))
decoder_states_input= [decoder_state_input_h, decoder_state_input_c]

dec_embd2 = decoder_embd(decoder_input)

decoder_output2,state_h2, state_c2 = decoder_lstm(dec_embd2, initial_state=decoder_states_input)
deccoder_states2= [state_h2, state_c2]

decoder_output2 = decoder_dense(decoder_output2)

decoder_model = Model(
                      [decoder_input]+decoder_states_input,
                      [decoder_output2]+ deccoder_states2)

"""#### Converting predicted numbers into text"""

def get_predicted_sentence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = mar_word_index['sos']

    # Sampling loop for a batch of sequences

    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index==0:
          break
        else:
         # convert max index number to marathi word
         sampled_char = mar_index_word[sampled_token_index]
        # aapend it ti decoded sent
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length or find stop token.
        if (sampled_char == 'eos' or len(decoded_sentence) >= 37):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

def get_marathi_sentence(sequence):
  sentence=""
  for i in sequence:
    if ((i != 0 and i != mar_word_index['sos']) and i != mar_word_index['eos']):
      sentence = sentence + mar_index_word[i]+' '
  return sentence

def get_eng_sent(sequence):
    sentence =''
    for i in sequence:
      if(i!=0):
        sentence = sentence + eng_index_word[i]+' '
    return sentence

for i in range(20):
  print("English sentence:",get_eng_sent(X_test[i]))
  print("Actual Marathi Sentence:",get_marathi_sentence(y_test[i]))
  print("Translated Marathi Sentence:",get_predicted_sentence(X_test[i].reshape(1,36))[:-4])
  print("\n")

