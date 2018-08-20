# -*- coding: utf-8 -*-
import os
import sys

import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

def train(train_path, val_path, test_path, batch_size=32, epochs=50, network='InceptionResNetV2', data_augmentation=False, mode='finetune', optimizer='Adadelta', fc=1, classes=2):
    '''
    Inputs:
        train_path: data path for train set (data should be stored like train/DR, train/Normal) 
        val_path: data path for validation set
        test_path: data path for test set
        batch_size: data sizes per step
        epochs: loop counts over whole train set
        network: {
            'InceptionResNetV2': fine-tune mode will train last 2 inception blocks
            'DenseNet201': fine-tune mode will train last Dense block
            'InceptionV3': fine-tune mode will train last 2 inception blocks
            'Xception'
            'NASNet'
            'MobileNetV2'
        }
        data_augmentation: whether to do data augmentation or not
        mode: {
            'retrain': randomly initialize all layers and retrain the whole model
            'finetune': train specified layers
            'transfer' train fc layer(s)
        }
        optimizer: {
            'Adadelta'
            'RMSprop'
        }
        fc: {
            1: only one fc layer at last
            2: include two fc layers at last
        }
        classes: category counts
    '''
    if mode == 'retrain':
        include_top = False
        weights = None
        pooling = 'avg'
    else:
        include_top = False
        weights = 'imagenet'
        pooling = 'avg'

    if network == 'DenseNet201':
        img_width, img_height = 224, 224
        base_model = applications.DenseNet201(include_top=include_top, weights=weights, pooling=pooling)
        # train last Dense Block
        if mode == 'finetune':
            trainable = False
            for layer in base_model.layers:
                if layer.name == 'conv5_block1_0_bn':
                    trainable = True
                layer.trainable = trainable

    if network == 'Xception':
        img_width, img_height = 299, 299
        base_model = applications.Xception(include_top=include_top, weights=weights, pooling=pooling)

    if network == 'InceptionV3':
        img_width, img_height = 299, 299
        base_model = applications.InceptionV3(include_top=include_top, weights=weights, pooling=pooling)
        # train top 2 inception blocks
        if mode == 'finetune':
            for layer in base_model.layers[:249]:
               layer.trainable = False
            for layer in base_model.layers[249:]:
               print(layer.name)
               layer.trainable = True

    if network == 'InceptionResNetV2':
        img_width, img_height = 299, 299
        base_model = applications.InceptionResNetV2(include_top=include_top, weights=weights, pooling=pooling)
        # train top 2 inception blocks
        if mode == 'finetune':
            trainable = False
            for layer in base_model.layers:
                if layer.name == 'conv2d_197':
                    trainable = True
                layer.trainable = trainable

    if network == 'NASNet':
        img_width, img_height = 331, 331
        base_model = applications.NASNetLarge(include_top=include_top, weights=weights, pooling=pooling)

    if network == 'MoblieNetV2':
        img_width, img_height = 224, 224
        base_model = applications.MobileNetV2(include_top=include_top, weights=weights, pooling=pooling)
   
    bottleneck = base_model.output
    if fc == 2:
        bottleneck = Dense(512, activation='relu')(bottleneck)
    predictions = Dense(classes, activation='softmax')(bottleneck)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    if mode == 'transfer':
        # train only the top layers (which were randomly initialized)
        # freeze all convolutional layers
        for layer in base_model.layers:
            layer.trainable = False

    if mode == 'retrain':
        # train a complete model
        for layer in base_model.layers:
            layer.trainable = True
    
    if optimizer == 'Adadelta':
        opt = optimizers.Adadelta()
    if optimizer == 'RMSprop':
        opt = optimizers.RMSprop()

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    if data_augmentation:
        # Initialize the train and test generators with data Augumentation 
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            fill_mode="nearest",
            zoom_range=0.3,
            width_shift_range=0.3,
            height_shift_range=0.3,
            rotation_range=30)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            fill_mode="nearest",
            zoom_range=0.3,
            width_shift_range=0.3,
            height_shift_range=0.3,
            rotation_range=30)

    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_height, img_width),
        batch_size=batch_size, 
        class_mode="categorical")

    validation_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(img_height, img_width),
        class_mode="categorical")

    test_generator = val_datagen.flow_from_directory(
        test_path,
        target_size=(img_height, img_width),
        class_mode="categorical")

    checkpoint = ModelCheckpoint("{}_custom.h5".format(network), monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

    model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint, early])

    model.evaluate_generator(test_generator)

if __name__ == '__main__':
    train_path = 'train'
    val_path = "val"
    test_path = "test"
    train(train_path, val_path, test_path)