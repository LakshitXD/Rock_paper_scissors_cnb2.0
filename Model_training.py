import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_model = InceptionV3(input_shape = (150,150,3), include_top=False, weights='imagenet')

for layer in base_model.layers:
	layer.trainable=False
	
base_model.summary()

last_layer = base_model.get_layer('mixed7')
last_output = last_layer.output
print('last layer output shape: ', last_layer.output_shape)

x = layers.Flatten()(last_output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(4, activation='softmax')(x)           

handModel = Model(base_model.input, x)

handModel.compile (optimizer = 'rmsprop', loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])


train = '/Train'
validation = '/Validation'

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory (train, batch_size=32, class_mode='categorical', target_size=(150,150))
validation_generator =  validation_datagen.flow_from_directory (validation, batch_size=32, class_mode='categorical', target_size=(150,150))

history = handModel.fit(
    train_generator,
    steps_per_epoch=56,
    epochs=100,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=21)

handModel.save('Model_4_classes.h5')



import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()

plt.show()