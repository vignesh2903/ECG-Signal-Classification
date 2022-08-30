# -*- coding: utf-8 -*-
"""
@author: Gaurav Prasanna
"""

import tensorflow as tf
import numpy as np
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses, Model
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

IMG_HEIGHT = 227
IMG_WIDTH = 227
IMG_CHANNELS = 3

img_folder = "ecgdataset/"


def create_dataset(img_folder):
    img_data_array = []
    class_name = []
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name


# extract the image array and class name
img_data, class_name = create_dataset(img_folder)
target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
target_val = [target_dict[class_name[i]] for i in range(len(class_name))]

X_train, X_test, y_train, y_test = train_test_split(np.array(img_data, np.float32),
                                                    np.array(list(map(int, target_val)),
                                                             np.float32),
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=0)

print("number of training examples =", X_train.shape[0])
print("number of test examples =", X_test.shape[0])
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

base_model = tf.keras.applications.vgg19.VGG19(weights='imagenet',
                                             include_top=False,
                                             input_shape=(227, 227, 3))
for layer in base_model.layers:
    layer.trainable = False

x = tf.keras.layers.Flatten()(base_model.output)
x = tf.keras.layers.Dense(1000, activation='relu')(x)
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

head_model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
head_model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                   metrics=['accuracy'])

history = head_model.fit(X_train, y_train, epochs=10,
                         batch_size=20, verbose=True)

head_model.evaluate(X_test, y_test)
y_pred = head_model.predict(X_test)

# report = classification_report(y_test, y_pred, target_names=target_val)
# print(report)









