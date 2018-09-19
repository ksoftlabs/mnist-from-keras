# Dense(300)
#  Epochs 5
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# Seed random to predictable results
np.random.seed(123)

# Load dataset and get x & y
dataset = pd.read_csv('all/train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert 28x28 matrix into 28x28 matrix with 1 channel
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

# Change range from 0-255 to 0 -1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# one hot encoding
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Define model as sequential
model = Sequential()

# Dense layers
model.add(Flatten())

model.add(Dense(784,activation='relu'))

model.add(Dense(300,activation='relu'))

model.add(Dense(100,activation='relu'))

model.add(Dense(100,activation='relu'))

model.add(Dense(200,activation='relu'))

# Softmax layer to categorize
model.add(Dense(10, activation='softmax'))

# COmpile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train,
          batch_size=32, epochs=10, verbose=1)

# Evaluate the model using test set
score = model.evaluate(X_test, Y_test, verbose=1)
print("Testing Loss : ", score[0], " Accuracy : ", score[1])

# Save model as JSON fle
model_json = model.to_json()
with open("model3.json", "w") as json_file:
    json_file.write(model_json)

# Save weights as h5
model.save_weights("model3.h5")
print("Saved model to disk")