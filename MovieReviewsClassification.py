from collections import Counter
import tensorflow as tf 
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing import sequence
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
def exploreData(training, test):

    training_data=training[0]
    training_labels = training[1]
    test_data = test[0]
    test_labels = test[1]
      
    #Collect the following important metrics that can help characterize your text classification problem

    #Number of samples
    print("Training entries: {}, training labels: {}".format(len    (training_data), len(training_labels)))

    # Number of classes
    print(np.unique(train_labels))
    print(np.unique(training_labels).size)

    #Number of samples per class
    numberOfPositives=0
    numberOfNegatives=0
    for x in training_labels:
        if x == 1:
            numberOfPositives = numberOfPositives + 1
        elif x == 0:
            numberOfNegatives =  numberOfNegatives + 1

    print("The number of positive reviews is " + str(numberOfPositives))

    print("The number of negative reviews is " + str(numberOfNegatives))

    #Number of words per sample: Median number of words in one sample
    sample_text = []
    count = 0
    for x in training_data:
        text = convert_integers_to_text(x)
        sample_text.append(text)
        count = count + 1
        if count == 100:
            break
        
    num_words = [len(s.split()) for s in sample_text]
    median = np.median(num_words)
    print(str(median))

   
    #Frequency distribution of words: Distribution showing the frequency (number of occurrences) of each word in the dataset

#https://github.com/google/eng-edu/blob/master/ml/guides/text_classification/explore_data.py

    #Distribution of sample length: Distribution showing the number of words per sample in the dataset


def convert_integers_to_text(word_integers):
    word_index = imdb.get_word_index()

    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["UNUSED"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    return ' '.join([reverse_word_index.get(i, '?') for i in word_integers])


imdb = keras.datasets.imdb

#Returns 2 tuples: https://keras.io/datasets/
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)
np.set_printoptions(threshold=np.nan)

"""
for row in train_data:
    print(row)
"""
#print(train_data.tolist())
#print(str(len(train_data)))
"""
print(type((train_data, train_labels)))
print(type(train_data))
print(len(train_data))
print(train_data.size)
print("The dimensions of the training data are " + str(train_data.shape))
"""

#Gather data

#Explore the data

print("train_data has a shape of " + str(train_data.shape))
#print(dir(imdb))
#print("FIRST MOVIE REVIEW")
#print(train_data[0])
#print("ALL LABELS")
#print(train_labels)
print(convert_integers_to_text(train_data[0]))
print(len(train_data[0]))
print(len(train_data[1]))

exploreData((train_data, train_labels), (test_data, test_labels))

#Choose a model

#Prepare your data
# https://keras.io/preprocessing/sequence/
train_data = pad_sequences(train_data, maxlen=500)
test_data = pad_sequences(test_data, maxlen=500)
print(len(train_data[0]))
print(len(test_data[0]))

print(train_data[0])

#Build, train, and evaluate your model
vocab_size = 10000
max_words = 500
"""
model = Sequential()
model.add(Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='sigmoid')) 
"""

"""
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
"""

"""
model = Sequential()
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
"""

""" DELETE
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
"""

""" MY MODEL
model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
"""

"""
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
"""

model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


#print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])


#fit the model
model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=2, batch_size=128, verbose=2)

scores = model.evaluate(test_data, test_labels, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print(model.summary())
#prediction_single = model.predict(test_data[0])

#print(predictions_single)

#Tune hyperparameters

#deploy your model
