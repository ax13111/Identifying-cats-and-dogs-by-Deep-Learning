#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:22:28 2023

@author: sunyenpeng
"""
import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import scipy
# Create directories for training and testing sets
#os.mkdir('train')
#os.mkdir('test')
# Copy the first 10,000 cat images to the training set
#cat_files = os.listdir('/Users/sunyenpeng/Desktop/python/kagglecatsanddogs_5340/PetImages/Cat')
#for file_name in cat_files[:10000]:
#    shutil.copy(f'/Users/sunyenpeng/Desktop/python/kagglecatsanddogs_5340/PetImages/Cat/{file_name}', f'/Users/sunyenpeng/Desktop/train/Cat/{file_name}')

# Copy the first 10,000 dog images to the training set
#dog_files = os.listdir('/Users/sunyenpeng/Desktop/python/kagglecatsanddogs_5340/PetImages/Dog')
#for file_name in dog_files[:10000]:
#    shutil.copy(f'/Users/sunyenpeng/Desktop/python/kagglecatsanddogs_5340/PetImages/Dog/{file_name}', f'/Users/sunyenpeng/Desktop/train/Dog/{file_name}')

# Copy the next 2,500 cat images to the testing set
#for file_name in cat_files[10000:12500]:
#    shutil.copy(f'/Users/sunyenpeng/Desktop/python/kagglecatsanddogs_5340/PetImages/Cat/{file_name}', f'/Users/sunyenpeng/Desktop/test/Cat/{file_name}')

# Copy the next 2,500 dog images to the testing set
#for file_name in dog_files[10000:12500]:
#    shutil.copy(f'/Users/sunyenpeng/Desktop/python/kagglecatsanddogs_5340/PetImages/Dog/{file_name}', f'/Users/sunyenpeng/Desktop/test/Dog/{file_name}')

#Building CNN model:
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
#Model compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
      

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/Users/sunyenpeng/Desktop/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    '/Users/sunyenpeng/Desktop/test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')
history = model.fit(
    train_generator,
    steps_per_epoch=2000,
    epochs=50, #Random Number
    validation_data=validation_generator,
    validation_steps=500)

#Model evaluation

model.evaluate(validation_generator)

#Loss Curve

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()















