from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('IMDB Dataset.csv')

max_words=10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

x_train[3]
y_train[3]

word_index = imdb.get_word_index()
word_index

reverse_word_index = dict()
for key, value in word_index.items():
    reverse_word_index[value] = key
    
for i in range(1, 21):
    print(i, '->', reverse_word_index[i])
