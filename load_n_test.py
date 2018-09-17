# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 14:07:25 2018

@author: ksoft
"""
from keras.models import model_from_json
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


#Load the JSON file with model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#Load weights from h5 file
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


#compile the loaded model
loaded_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



#Load dataset and seperate X n Y
dataset = pd.read_csv('all/train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

#Seperate Test and Train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Reshape to 28x28 matrix with 1 channel
X_train = X_train.reshape(X_train.shape[0],1,28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)


#Change range from 0-255 to 0 -1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#One hot encoding
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#Test the model
score = loaded_model.evaluate(X_test, Y_test, verbose=1)
print("Testing Loss : ",score[0]," Accuracy : ",score[1]*100,"%")