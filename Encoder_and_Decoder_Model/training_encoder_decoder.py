

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

df= pd.read_csv("/content/drive/MyDrive/ML/rnn/Machine_Translation/Data/cleaned.csv")

df.tail()

"""**Note** this data was cleaned at notebook of EDA_And_Cleaning_Text

# Prepare dataset for encoder decoder model
## Encoder:-
* Here first we will convert text into numbers
* [WordEmbedding](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/):- Then embedding is very important layer for beause it will convert the input word-`numbers into more dimension of vectors which will have semantic information words means beause of this we can know which words are similar or near to each other.`
* We will not take output of timestamps of encoder beause it will be like one to one mapping so we will just take selt states of encoder as context vector.

## Decoder:-
* First most important thing is we have to add special tokens in each target language at start SOS and EOS at end reason of this is `The length of translated sentence might not be same as other language so it is to tell model where is start and end of sentence.`
* When building model we will provide initial state of model as context vector recived from Encoder.

#### Fisrt add eos and sos tokens
* SOS = Start Of String
* EOS = End Of String
"""

df.Marathi = df.Marathi.apply(lambda x: 'sos '+ x +' eos')

"""### create vocabulary of english and marathi words"""

eng_vocab= set()
for sent in df.English:
    for word in sent.split():
        if word not in eng_vocab:
            eng_vocab.add(word)

mar_vocab= set()
for sent in df.Marathi:
    for word in sent.split():
        if word not in mar_vocab:
            mar_vocab.add(word)

len(eng_vocab), len(mar_vocab),

# for zero padding add 1 in them
ENG_VOCAB_SIZE= len(eng_vocab)+1
MAR_VOCAB_SIZE= len(mar_vocab)+1
print(ENG_VOCAB_SIZE)
print(MAR_VOCAB_SIZE)

"""### Create dictionary for words and their indexes then we can convert text into numbers

#### First we need sorted words list
"""

eng_words = sorted(list(eng_vocab))
mar_words = sorted(list(mar_vocab))

"""#### Word to number"""

# create english and marathi dicts
eng_word_index = dict((w, i) for i, w in enumerate(eng_words))
mar_word_index = dict((w, i) for i, w in enumerate(mar_words))

mar_word_index

"""#### Number to word
*  we will need this one at time of creating text from predicted values
"""

eng_index_word = dict((i, w) for i, w in enumerate(eng_words))
mar_index_word = dict((i,w) for i, w in enumerate(mar_words))

mar_index_word

"""### Train test split"""

X_train, X_test, y_train, y_test= train_test_split(df.English, df.Marathi, test_size=0.1, random_state=0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

"""## Create data generator
* if we create array of 3d shape with our vocab size this will give us out of memmory error
* And it is always best to use batches to train it will make process faster
* Insted of passing all data in model which may run out of memory we create data generator which will create data batches at time of training

#### prepare input for encoder decoder [refer](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)
* Turn the sentences into 3 Numpy arrays, encoder_input_data, decoder_input_data, decoder_target_data:
>* encoder_input_data is a 3D array of shape (num_pairs,max_english_sentence_length, num_english_characters) containing a one-hot vectorization of the English sentences.
>* decoder_input_data is a 3D array of shape (num_pairs, max_french_sentence_length, num_french_characters) containg a one-hot vectorization of the French sentences.
>* decoder_target_data is the same as decoder_input_data but offset by one timestep. decoder_target_data[:, t, :] will be the same as decoder_input_data[:, t + 1, :].
* 2) Train a basic LSTM-based Seq2Seq model to predict decoder_target_data given encoder_input_data and decoder_input_data. Our model uses teacher forcing.
* 3) Decode some sentences to check that the model is working (i.e. turn samples from encoder_input_data into corresponding samples from decoder_target_data).

* We use a technique called “Teacher Forcing” wherein the input at each time step is given as the actual output (and not the predicted output) from the previous time step.

#### Before we go ahead lets define some things we need for data generator
"""

### Get lengths of each sentence in list
eng_len_list=df.English.apply(lambda x: len(x.split())).to_list()

mar_len_list=df.Marathi.apply(lambda x: len(x.split())).to_list()

# get max length
np.max(mar_len_list), np.max(eng_len_list)

BATCH_SIZE= 64
max_eng_len =  np.max(eng_len_list)
max_mar_len =  np.max(mar_len_list)

max_eng_len, max_mar_len

"""## Get data generator fuction from [keras](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly) team
* also visit [here](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) to see how keras created this input's for encoder and decoder
"""

def data_batch_generator(x, y, batch_size=BATCH_SIZE):
    while True:
        for i in range(0, len(x), batch_size):
            encoder_input_data = np.zeros((batch_size,max_eng_len ), dtype='float32')
            decoder_input_data = np.zeros((batch_size, max_mar_len), dtype='float32')
            # one hot encoded target data beause dense layer with softmax will give only one output at a time
            decoder_target_data = np.zeros((batch_size, max_mar_len, MAR_VOCAB_SIZE), dtype='float32' )
            for j, (eng_text, mar_text) in enumerate(zip(x[i:i+batch_size], y[i:i+batch_size])):
                for t , word in enumerate(eng_text.split()):
                    encoder_input_data[j,t] = eng_word_index[word]
                for t, word in enumerate(mar_text.split()):
                    if t < len(mar_text.split()) - 1:
                        decoder_input_data[j,t]= mar_word_index[word]
                    if t>0:
                        # This is decoder target output which is one step ahead of decoder input
                        # it does not have EOS token
                        decoder_target_data[j,t-1, mar_word_index[word]] = 1.
            yield ([encoder_input_data, decoder_input_data], decoder_target_data)

"""# LSTM Encoder Decoder Model"""

from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding
from tensorflow.keras import Input, Model

# Eoncoder
encoder_input = Input(shape=(None, ))
encoder_embd = Embedding(ENG_VOCAB_SIZE,100, mask_zero=True)(encoder_input)
encoder_lstm = LSTM(100, return_state=True)
encoder_output,state_h, state_c = encoder_lstm(encoder_embd)

## Now take only states and create context vector
encoder_states= [state_h, state_c]

# Decoder
decoder_input = Input(shape=(None,))
# For zero padding we have added +1 in marathi vocab size
decoder_embd = Embedding(MAR_VOCAB_SIZE, 100, mask_zero=True)
decoder_embedding= decoder_embd(decoder_input)
decoder_lstm = LSTM(100, return_state=True,return_sequences=True )
# just take output of this decoder dont need self states
decoder_outputs, _, _= decoder_lstm(decoder_embedding, initial_state=encoder_states)
# here this is going to predicct so we can add dense layer here
# here we want to convert predicted numbers into probability so use softmax
decoder_dense= Dense(MAR_VOCAB_SIZE, activation='softmax')
# We will again feed predicted output into decoder to predict its next word
decoder_outputs = decoder_dense(decoder_outputs)

model1 = Model([encoder_input, decoder_input], decoder_outputs)

from tensorflow.keras.utils import  plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
plot_model(model2,show_shapes=True)

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

checkpoint = ModelCheckpoint("/content/drive/MyDrive/rnn/machine_translation/Encoder_Decoder/model_checkpoints/", monitor='val_accuracy')

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

callbacks_list = [checkpoint, early_stopping]

"""#### **IMP NOTE** - here to avoid unknown samples need to use sptes per epoch else model will fit for infinite samples"""

steps_per_epoch= np.ceil(len(X_train)/BATCH_SIZE)
steps_per_epoch_val = np.ceil(len(X_train)/BATCH_SIZE)

steps_per_epoch

EPOCHS= 30 #@param {type:'slider',min:10,max:100, step:10 }
EPOCHS

history1= model1.fit(data_batch_generator(X_train,y_train),
                       epochs=EPOCHS,
                       steps_per_epoch= steps_per_epoch,
                     validation_data=data_batch_generator(X_test, y_test, BATCH_SIZE),
                       validation_steps=steps_per_epoch_val,
                     callbacks=callbacks_list)

model1.save_weights('/content/drive/MyDrive/rnn/machine_translation/Encoder_Decoder/saved_models/model1_weights.h5')

model1.load_weights('/content/drive/MyDrive/rnn/machine_translation/Encoder_Decoder/saved_models/model1_weights.h5')

"""# Model2
Now we will try to improve its accurcy with changing some units
"""

# Eoncoder
encoder_input = Input(shape=(None, ))
encoder_embd = Embedding(ENG_VOCAB_SIZE,1000, mask_zero=True)(encoder_input)
encoder_lstm = LSTM(250, return_state=True)
encoder_output,state_h, state_c = encoder_lstm(encoder_embd)

## Now take only states and create context vector
encoder_states= [state_h, state_c]

# Decoder
decoder_input = Input(shape=(None,))
# For zero padding we have added +1 in marathi vocab size
decoder_embd = Embedding(MAR_VOCAB_SIZE, 1000, mask_zero=True)
decoder_embedding= decoder_embd(decoder_input)
decoder_lstm = LSTM(250, return_state=True,return_sequences=True )
# just take output of this decoder dont need self states
decoder_outputs, _, _= decoder_lstm(decoder_embedding, initial_state=encoder_states)
# here this is going to predicct so we can add dense layer here
# here we want to convert predicted numbers into probability so use softmax
decoder_dense= Dense(MAR_VOCAB_SIZE, activation='softmax')
# We will again feed predicted output into decoder to predict its next word
decoder_outputs = decoder_dense(decoder_outputs)

model2 = Model([encoder_input, decoder_input], decoder_outputs)

from tensorflow.keras.utils import  plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
plot_model(model2,show_shapes=True)

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

checkpoint = ModelCheckpoint("/content/drive/MyDrive/rnn/machine_translation/Encoder_Decoder/model_checkpoints/model2/", monitor='val_accuracy')

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

callbacks_list = [checkpoint, early_stopping]

steps_per_epoch= np.ceil(len(X_train)/BATCH_SIZE)
steps_per_epoch_val = np.ceil(len(X_train)/BATCH_SIZE)

EPOCHS= 30 #@param {type:'slider',min:10,max:100, step:10 }
EPOCHS

history2= model2.fit(data_batch_generator(X_train,y_train),
                       epochs=EPOCHS,
                       steps_per_epoch= steps_per_epoch,
                     validation_data=data_batch_generator(X_test, y_test, BATCH_SIZE),
                       validation_steps=steps_per_epoch_val,
                     callbacks=callbacks_list)

model2.save_weights('/content/drive/MyDrive/rnn/machine_translation/Encoder_Decoder/saved_models/2_model_weights.h5')

model2.load_weights('/content/drive/MyDrive/rnn/machine_translation/Encoder_Decoder/saved_models/2_model_weights.h5')

"""## Inference model
*  As we trained our enoder decoder do same for prdiction means apply encoder on input sent and applying decoder on target sent
"""

encoder_model = Model(encoder_input, encoder_states)

decoder_state_input_h = Input(shape=(250,))
decoder_state_input_c= Input(shape=(250,))
decoder_states_input= [decoder_state_input_h, decoder_state_input_c]

dec_embd2 = decoder_embd(decoder_input)

decoder_output2,state_h2, state_c2 = decoder_lstm(dec_embd2, initial_state=decoder_states_input)
deccoder_states2= [state_h2, state_c2]

decoder_output2 = decoder_dense(decoder_output2)

decoder_model = Model(
                      [decoder_input]+decoder_states_input,
                      [decoder_output2]+ deccoder_states2)

"""# To predict we have to encoder text first then pass than to decoder we can get predicted values"""

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
        # convert max index number to marathi word
        sampled_char = mar_index_word[sampled_token_index]
        # aapend it ti decoded sent
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length or find stop token.
        if (sampled_char == 'eos' or len(decoded_sentence) > 50):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

test_gen= data_batch_generator(X_test,y_test,batch_size=1)

Actual_test_sent = X_test.to_list()
Actual_test_trans= y_test.to_list()

test_inputs=[]
test_outputs=[]

from tqdm import tqdm
for (input, output),_ in tqdm(iter(test_gen)):
    test_inputs.append(input)
    test_outputs.append(output)

test_inputs[0]

print("English Senteces:", Actual_test_sent[0])
print("Actual Marathi Sentence:", Actual_test_trans[0][:-4])
print("Predicted Marathi Translation:", get_predicted_sentence(test_inputs[0])[:-4])

print("English Senteces:", Actual_test_sent[1])
print("Actual Marathi Sentence:", Actual_test_trans[1][:-4])
print("Predicted Marathi Translation:", get_predicted_sentence(test_inputs[1])[:-4])

print("English Senteces:", Actual_test_sent[3])
print("Actual Marathi Sentence:", Actual_test_trans[3][:-4])
print("Predicted Marathi Translation:", get_predicted_sentence(test_inputs[3])[:-4])

print("English Senteces:", Actual_test_sent[10])
print("Actual Marathi Sentence:", Actual_test_trans[10][:-4])
print("Predicted Marathi Translation:", get_predicted_sentence(test_inputs[10])[:-4])

print("English Senteces:", Actual_test_sent[50])
print("Actual Marathi Sentence:", Actual_test_trans[50][:-4])
print("Predicted Marathi Translation:", get_predicted_sentence(test_inputs[50])[:-4])

print("English Senteces:", Actual_test_sent[100])
print("Actual Marathi Sentence:", Actual_test_trans[100][4:-4])
print("Predicted Marathi Translation:", get_predicted_sentence(test_inputs[100])[:-4])

print("English Senteces:", Actual_test_sent[89])
print("Actual Marathi Sentence:", Actual_test_trans[89][4:-4])
print("Predicted Marathi Translation:", get_predicted_sentence(test_inputs[89])[:-4])

print("English Senteces:", Actual_test_sent[77])
print("Actual Marathi Sentence:", Actual_test_trans[77][4:-4])
print("Predicted Marathi Translation:", get_predicted_sentence(test_inputs[77])[:-4])

print("English Senteces:", Actual_test_sent[123])
print("Actual Marathi Sentence:", Actual_test_trans[123][4:-4])
print("Predicted Marathi Translation:", get_predicted_sentence(test_inputs[123])[:-4])

print("English Senteces:", Actual_test_sent[165])
print("Actual Marathi Sentence:", Actual_test_trans[165][:-4])
print("Predicted Marathi Translation:", get_predicted_sentence(test_inputs[165])[:-4])

print("English Senteces:", Actual_test_sent[177])
print("Actual Marathi Sentence:", Actual_test_trans[177][:-4])
print("Predicted Marathi Translation:", get_predicted_sentence(test_inputs[177])[:-4])

