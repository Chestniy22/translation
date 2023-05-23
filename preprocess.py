from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('IMDB Dataset.csv')

word_index = imdb.get_word_index()
word_index

reverse_word_index = dict()
for key, value in word_index.items():
    reverse_word_index[value] = key
    
for i in range(1, 21):
    print(i, '->', reverse_word_index[i])

    index = 3
message = ''
for code in x_train[index]:
    word = reverse_word_index.get(code - 3, '?')
    message += word + ' '
message

y_train[index]

maxlen = 200

x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')

x_train[3]

y_train[3]
