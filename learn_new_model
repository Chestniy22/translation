from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

max_words=10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)


# Вычисляем количество элементов, которые нужно удалить
num_to_remove = int(0.1 * len(x_train))

# Удаляем элементы из обучающей выборки и соответствующих меток классов
x_train_new, x_val = np.split(x_train, [len(x_train)-num_to_remove])
y_train_new, y_val = np.split(y_train, [len(y_train)-num_to_remove])

# Вычисляем количество элементов, которые нужно удалить из тестовой выборки и соответствующих меток классов
num_to_remove = int(0.1 * len(x_test))

# Удаляем элементы из тестовой выборки и соответствующих меток классов
x_test_new, x_val_test = np.split(x_test, [len(x_test)-num_to_remove])
y_test_new, y_val_test = np.split(y_test, [len(y_test)-num_to_remove])

# Сохраняем новые выборки в файлы
np.save('x_train_new.npy', x_train_new)
np.save('y_train_new.npy', y_train_new)
np.save('x_val.npy', x_val)
np.save('y_val.npy', y_val)
np.save('x_test_new.npy', x_test_new)
np.save('y_test_new.npy', y_test_new)
np.save('x_val_test.npy', x_val_test)
np.save('y_val_test.npy', y_val_test)

