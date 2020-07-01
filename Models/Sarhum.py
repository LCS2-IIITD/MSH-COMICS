
from scipy import spatial



from attention import AttentionLayer,attention


import numpy as np
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
import subprocess

def install(name):
    subprocess.call(['pip', 'install', name])
install('keras_self_attention')    

from keras_self_attention import SeqSelfAttention as Attention1

import keras
import keras.utils
from keras import utils as np_utils
from keras.utils import to_categorical

from tqdm import tqdm
tqdm.pandas()

import re
import h5py
import json
import pickle
import random
from PIL import Image
from pickle import load

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display
from keras.backend import clear_session
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from nltk.translate.bleu_score import corpus_bleu
from keras.utils import to_categorical, plot_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add,TimeDistributed
from keras.models import Model, Sequential, model_from_json, load_model

from keras.layers import  Bidirectional

from keras import initializers, regularizers, constraints, optimizers, layers

from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Embedding,multiply,Lambda,Convolution1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import collections
import re
import string

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pickle
f=open('/content/drive/My Drive/Sarcasm/Final_dataset3.p','rb')
d_final3=pickle.load(f)

import pickle
f=open('/content/drive/My Drive/Sarcasm/embedding_matrix.p','rb')
embedding_matrix=pickle.load(f)
f=open('/content/drive/My Drive/Sarcasm/Audio_features.p','rb')
emb=pickle.load(f)

MAX_NB_WORDS=40000
MAX_SEQUENCE_LENGTH=128

text=d_final3['text']
tokenizer=Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(text)
sequences=tokenizer.texts_to_sequences(text)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token=True)
tokenizer.fit_on_texts(text)
data = np.zeros((len(text),MAX_SEQUENCE_LENGTH),dtype='int32')

labels1=[]
label_index1={}
for label in d_final3['Sarcasm']:
  labelid=len(label_index1)
  label_index1[label]=labelid
  labels1.append(label)

print(len(labels1))

labels2=[]
label_index2={}
for label in d_final3['Humour']:
  labelid=len(label_index1)
  label_index2[label]=labelid
  labels2.append(label)

print(len(labels2))

embedding=[]
for i in range(len(emb)):
  embedding.append(emb[i])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data=pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)
labels1 = np.asarray(labels1)
labels2 = np.asarray(labels2)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels1.shape)

import os
f=open('d_features.p','rb')
data=pickle.load(f)

test_data=data[14000:15576]
test_label1=labels1[14000:15576]
test_label2=labels2[14000:15576]
test_embedding=embedding[14000:15576]
test_data.shape,test_label1.shape,test_label2.shape,len(test_embedding)

data=data[0:14000]
labels1=labels1[0:14000]
labels2=labels2[0:14000]
embedding=embedding[0:14000]
data.shape,labels1.shape,len(embedding)

EMBEDDING_DIM=300
print(embedding_matrix.shape)
embedding_layer=Embedding(len(word_index)+1,EMBEDDING_DIM,weights=[embedding_matrix],trainable=False,name='embedding',mask_zero=True,input_shape=(6,128))

VALIDATION_SPLIT=0.2
indices = np.arange(data.shape[0])
data = data[indices]
labels1 = labels1[indices]
labels2= labels2[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

train_x1 = data[:-nb_validation_samples]
train_x2 = embedding[:-nb_validation_samples] 
train_y1 = labels1[:-nb_validation_samples]
train_y2 = labels2[:-nb_validation_samples]
dev_x1 = data[-nb_validation_samples:]
dev_x2 = embedding[-nb_validation_samples:]
dev_y1 = labels1[-nb_validation_samples:]
dev_y2 = labels2[-nb_validation_samples:]

train_x2=np.array(train_x2)
dev_x2=np.array(dev_x2)

train_x2=train_x2.reshape((11200,128,1))
train_x2.shape

dev_x2=dev_x2.reshape((2800,128,1))
dev_x2.shape

max_sentences=10
max_words=128
word_encoding_dim=300
sentence_encoding_dim=128

from keras.backend import manual_variable_initialization
manual_variable_initialization(True)

from keras.layers import Layer
import keras.backend as K

import keras
from keras.layers import (
    Dense, GRU, TimeDistributed, Input,
    Embedding, Bidirectional, Lambda
)
from keras.models import Model
def build_word_encoder(max_words, embedding_matrix, encoding_dim=128):
  vocabulary_size = embedding_matrix.shape[0]
  embedding_dim = embedding_matrix.shape[1]
  sentence_input = Input(shape=(max_words,), dtype='int32')
  embedded_sentences = embedding_layer(sentence_input)
  word_embed,sample = Attention1(attention_width=3,attention_activation='sigmoid',name='attention',return_attention=True)(embedded_sentences)
  encoded_sentences = Bidirectional(GRU(int(encoding_dim / 2), return_sequences=True))(word_embed)
  return Model(inputs=[sentence_input], outputs=[encoded_sentences], name='word_encoder')
def build_sentence_encoder(self, max_sentences, summary_dim, encoding_dim=300):
  text_input = Input(shape=(max_sentences, summary_dim))
  encoded_sentences = Bidirectional(GRU(int(encoding_dim / 2), return_sequences=True))(text_input)
  return Model(inputs=[text_input], outputs=[encoded_sentences], name='sentence_encoder')
in_tensor = Input(shape=(max_sentences, max_words))
word_encoder = build_word_encoder(max_words, embedding_matrix, word_encoding_dim)
word_rep = TimeDistributed(word_encoder, name='word_encoder')(in_tensor)
sentence_rep = TimeDistributed(LSTM(128))(word_rep)
sentence_rep1 = TimeDistributed(Attention1(attention_width=3,attention_activation='sigmoid',name='attention'))(word_rep)
doc_rep = build_sentence_encoder(max_sentences,word_encoding_dim,sentence_encoding_dim)(sentence_rep)
doc_summary = AttentionLayer(name='sentence_attention')(doc_rep)    
layer2=LSTM(128,return_sequences=True)(doc_rep)
'''encodings=TimeDistributed(LSTM(128))(layer2)'''
print(sentence_rep.shape)
word_att,sample = Attention1(attention_width=3,attention_activation='sigmoid',name='attention',return_attention=True)(layer2)
fe_inputs = Input(shape=(128,1))
a=Convolution1D(128,kernel_size=128,padding='same')(fe_inputs)
b=Convolution1D(128,kernel_size=128,padding='same')(a)
fe_layer1,ha,cella = LSTM(128,return_sequences=True,return_state=True,dropout=0.4)(b)
att_audio=AttentionLayer(name='audio_attention')(fe_layer1)
Ha=concatenate([ha,att_audio])
seq_layer2,ht,cell=LSTM(128,return_sequences=True ,return_state=True,dropout=0.3)(word_att)
att_out=AttentionLayer(name='sentence_attention')(seq_layer2)
print(att_out.shape)
Ht=concatenate([ht,att_out])
HAT=concatenate([ht,ha])
HAT_T=Dense(units=Ht.shape[-1],activation=K.relu)(HAT)
HT=multiply([Ht,HAT_T])
HA=multiply([Ha,HAT_T])
all_input=concatenate([HAT,HA,HT])
decoder_layer1 = Dense(128, activation='relu')(all_input)
decoder_layer2 = Dense(128, activation='relu')(all_input)
outputs1=Dense(1,activation='sigmoid')(decoder_layer1)
outputs2=Dense(1,activation='sigmoid')(decoder_layer2)
model_sh2= Model(inputs=[fe_inputs,in_tensor], outputs=[outputs1,outputs2])
model_sh2.compile(loss='binary_crossentropy', optimizer='adam',metrics=["accuracy"])
model_sh2.summary()

model_encoding=Model(inputs=[in_tensor],outputs=[att_out])

train_x1.shape

model_name = '15khumour+sarasmmodel2(new).h5'
n_epochs =15
batch_size =32 
checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = model_sh2.fit([train_x2,train_x1], [train_y2,train_y1], epochs=n_epochs, batch_size=batch_size, validation_data=([dev_x2,dev_x1], [dev_y2,dev_y1]), callbacks=[checkpoint])
model_sh2.save('15k211mode3.h5(new)')

