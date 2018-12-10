import tensorflow as tf
from tensorflow import keras

imdb = keras.datasets.imdb                                                                          

#Returns 2 tuples: https://keras.io/datasets/
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)
np.set_printoptions(threshold=np.nan)

