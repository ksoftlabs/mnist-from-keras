# Conv2D (32,3,3) -> Conv2D (32,3,3) -> MaxPooling2D(2,2) -> DropOut (0.25)  -> Flatten -> Dense (256) -> DropOut(0.5) -> Dense (10)#
# Epochs 5
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

# Add a convolutional layer - 32 filters and 3x3 matrix
model.add(Conv2D(filters=32, kernel_size=[3, 3], activation='relu', input_shape=(28, 28,1)))

# Add a convolutional layer - 32 filters and 3x3 matrix
model.add(Conv2D(filters=32, kernel_size=[3, 3], activation='relu'))

# MaxPool the resulting matrix
model.add(MaxPooling2D(pool_size=(2, 2)))

# A drop out layer to prevent overfitting
model.add(Dropout(0.25))

# Flattern the 2D matrix
model.add(Flatten())

# Fully connected layer with 256 nodes
model.add(Dense(256, activation='relu'))

# A drop out layer to prevent overfitting
model.add(Dropout(0.5))

# Softmax layer to categorize
model.add(Dense(10, activation='softmax'))

# COmpile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train,
          batch_size=32, epochs=5, verbose=1)

# Evaluate the model using test set
score = model.evaluate(X_test, Y_test, verbose=1)
print("Testing Loss : ", score[0], " Accuracy : ", score[1])

# # Save model as JSON fle
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
#
# # Save weights as h5
# model.save_weights("model.h5")
# print("Saved model to disk")