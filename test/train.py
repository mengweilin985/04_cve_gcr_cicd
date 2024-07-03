import warnings
warnings.filterwarnings("ignore")

import os
import cv2

def load_images(directory, image_size = (512, 512)):
    X_values = []
    y_values = []

    for each_folder in os.listdir(directory):
        if each_folder != '.DS_Store':
            sub_path = f'{directory}/{each_folder}'

            for each_image in os.listdir(sub_path):
                    if each_image != '.DS_Store':
                        image = cv2.cvtColor(cv2.resize(cv2.imread(f'{sub_path}/{each_image}'),image_size), cv2.COLOR_BGR2GRAY)
                        X_values.append(image)
                        y_values.append(int(each_folder[0]))
    return X_values, y_values

IMAGE_SIZE = (28, 28)

X_values, y_values = load_images(directory = './data', image_size = IMAGE_SIZE)

# print(f'{len(X_values)} images loaded into X_values')
# print(f'{len(y_values)} images loaded into y_values')

import numpy as np

X_values = np.array(X_values)
#print(X_values[:1])

y_values = np.array(y_values)
#print(y_values)

X_values = X_values / 255
#print(X_values[:1])

X_values = X_values.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
#print(X_values.shape)

import matplotlib.pyplot as plt
import random

def plot(num_columns, num_rows):
    _, axis = plt.subplots(ncols = num_columns, nrows = num_rows)

    for each_row in range(num_rows):
        for each_column in range(num_columns):
            index = random.randint(0, len(X_values) - 1)

            axis[each_row][each_column].imshow(X_values[index], cmap = 'gray')
            axis[each_row][each_column].set_title(y_values[index])
            axis[each_row][each_column].axis('off')

    return None
    
plot(num_columns = 5, num_rows = 2)

from sklearn.model_selection import train_test_split

X_OTHER, X_test, y_OTHER, y_test = train_test_split(X_values, y_values, test_size = 0.1, stratify = y_values, random_state = 42)
# print(f'Shape of X_test: {X_test.shape}')
# print(f'Shape of y_test: {y_test.shape}')

X_train, X_val, y_train, y_val = train_test_split(X_OTHER, y_OTHER, test_size = 0.3, stratify = y_OTHER, random_state = 42)
# print(f'Shape of X_train: {X_train.shape}')
# print(f'Shape of y_train: {y_train.shape}\n')
# print(f'Shape of X_val: {X_val.shape}')
# print(f'Shape of y_val: {y_val.shape}')

from keras.models import Sequential

from keras.layers import (Conv2D, Flatten, Dense)

num_classes = len(np.unique(y_values))

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],1)))
model.add(Flatten())
model.add(Dense(num_classes, activation ='softmax'))

model.summary()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

config = vars(vars(model)['_compile_config'])['config']

print(f"Loss: {config['loss']}")
print(f"Optimiser: {config['optimizer']}")
print(f"Metrics: {config['metrics']}")

history = model.fit(X_train, y_train, epochs = 10, validation_data = (X_val, y_val), verbose = 1)

plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.plot(history.history['accuracy'], label = 'Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

model.save("../nn.h5")
##############################################################################
import keras
new_model = keras.models.load_model("../nn.h5")

y_pred = new_model.predict(X_test).argmax(axis = 1)
print(y_pred)

from sklearn.metrics import classification_report

print(classification_report(y_true = y_test, y_pred = y_pred))


##############################################################################

