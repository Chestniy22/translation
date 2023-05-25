from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report
import joblib

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

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(maxlen,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

history = model.fit(x_train, 
                    y_train, 
                    epochs=25,
                    batch_size=128,
                    validation_split=0.1)
joblib.dump(model,'model1.joblib')
target_names = ['negative', 'positive']
start_time = time.time()
y_pred = model.predict(x_test)
end_time = time.time()
conf_matrix = classification_report(y_pred.round(),y_test)
report = classification_report(y_test, y_pred.round(), target_names=target_names)

with open('results.txt', 'w') as f:
    f.write('Classification report:\n{}\n\n'.format(report))
    f.write('time on test data: {}%\n'.format(end_time))
    
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(maxlen,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

history = model.fit(x_train, 
                    y_train, 
                    epochs=25,
                    batch_size=128,
                    validation_split=0.1)
joblib.dump(model,'model2.joblib')
target_names = ['negative', 'positive']
start_time = time.time()
y_pred = model.predict(x_test)
end_time = time.time()
conf_matrix = classification_report(y_pred.round(),y_test)
report = classification_report(y_test, y_pred.round(), target_names=target_names)

with open('results2.txt', 'w') as f:
    f.write('Classification report:\n{}\n\n'.format(report))
    f.write('time on test data: {}%\n'.format(end_time))
