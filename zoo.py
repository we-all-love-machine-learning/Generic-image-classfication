# -*- coding: utf-8 -*-
import os
import sys

import keras
import numpy as np

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import multi_gpu_model
from keras.models import load_model

def train(train_path, val_path, test_path, batch_size=32, epochs=50, network='InceptionResNetV2', data_augmentation=True, mode='finetune', optimizer='Adadelta', fc=1, classes=5, gpu=1):
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
            'ResNet50': According to https://arxiv.org/pdf/1805.08974.pdf, it is most suitable for transfer learning?
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
        from keras.applications.densenet import preprocess_input
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
        from keras.applications.xception import preprocess_input
        img_width, img_height = 299, 299
        base_model = applications.Xception(include_top=include_top, weights=weights, pooling=pooling)

    if network == 'InceptionV3':
        from keras.applications.inception_v3 import preprocess_input
        img_width, img_height = 299, 299
        base_model = applications.InceptionV3(include_top=include_top, weights=weights, pooling=pooling)
        # train top 2 inception blocks
        if mode == 'finetune':
            for layer in base_model.layers[:249]:
               layer.trainable = False
            for layer in base_model.layers[249:]:
               #print(layer.name)
               layer.trainable = True

    if network == 'InceptionResNetV2':
        from keras.applications.inception_resnet_v2 import preprocess_input
        img_width, img_height = 299, 299
        base_model = applications.InceptionResNetV2(include_top=include_top, weights=weights, pooling=pooling)
        # train top 1 inception blocks
        if mode == 'finetune':
            trainable = True
            for layer in base_model.layers:
                #print(layer.name)
                if layer.name == 'conv2d_9':
                    trainable = False
                if layer.name == 'conv2d_201':
                    trainable = True  
                layer.trainable = trainable

    if network == 'NASNet':
        from keras.applications.nasnet import preprocess_input
        img_width, img_height = 331, 331
        base_model = applications.NASNetLarge(include_top=include_top, weights=weights, pooling=pooling)

    if network == 'MoblieNetV2':
        from keras.applications.mobilenetv2 import preprocess_input
        img_width, img_height = 224, 224
        base_model = applications.MobileNetV2(include_top=include_top, weights=weights, pooling=pooling)

    if network == 'ResNet50':
        from keras.applications.resnet50 import preprocess_input
        img_width, img_height = 224, 224
        base_model = applications.ResNet50(include_top=include_top, weights=weights, pooling=pooling)
   
    bottleneck = base_model.output
    if fc == 2:
        bottleneck = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.001))(bottleneck)
    predictions = Dense(classes, kernel_regularizer=keras.regularizers.l2(l=0.001), activation='softmax', bias_regularizer=keras.regularizers.l2(l=0.001))(bottleneck)
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
    if optimizer == 'Adam':
        opt = optimizers.Adam()
    if optimizer == 'RMSprop':
        opt = optimizers.RMSprop(lr=0.005, rho=0.9, epsilon=1.0, decay=0.94) 

    if gpu > 1:
        batch_size *= gpu
        model = multi_gpu_model(model, gpus=gpu)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    if data_augmentation:
        # Initialize the train and test generators with data Augumentation 
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            horizontal_flip=True,
            fill_mode="nearest",
            zoom_range=0.3,
            width_shift_range=0.3,
            height_shift_range=0.3,
            rotation_range=30)
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

    else:
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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

    checkpoint = ModelCheckpoint("{}_{}_{}.h5".format(network, mode, optimizer), monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

    model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint, early])

    score = model.evaluate_generator(test_generator)

    print(score)
    
def evaluate(model_path, test_path):
    model = load_model(model_path)

    if 'Adadelta' in model_path:
        opt = optimizers.Adadelta()
    if 'RMSprop' in model_path:
        opt = optimizers.RMSprop()
    if 'Adam' in model_path:
        opt = optimizers.Adam()

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    if 'DenseNet201' in model_path:
        from keras.applications.densenet import preprocess_input
        img_width, img_height = 224, 224

    if 'Xception' in model_path:
        from keras.applications.xception import preprocess_input
        img_width, img_height = 299, 299

    if 'InceptionV3' in model_path:
        from keras.applications.inception_v3 import preprocess_input
        img_width, img_height = 299, 299

    if 'InceptionResNetV2' in model_path:
        from keras.applications.inception_resnet_v2 import preprocess_input
        img_width, img_height = 299, 299

    if 'NASNet' in model_path:
        from keras.applications.nasnet import preprocess_input
        img_width, img_height = 331, 331

    if 'MoblieNetV2' in model_path:
        from keras.applications.mobilenetv2 import preprocess_input
        img_width, img_height = 224, 224

    if network == 'ResNet50':
        from keras.applications.resnet50 import preprocess_input
        img_width, img_height = 224, 224

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_height, img_width),
        class_mode="categorical")

    # [loss, accuracy]
    score = model.evaluate_generator(test_generator)

    print(score)

if __name__ == '__main__':
    train_path = 'train_stages/train'
    val_path = "train_stages/val"
    test_path = "train_stages/test"
    model_path = 'Xception_transfer.h5'
    train(train_path, val_path, test_path, batch_size=24)
    evaluate(model_path, test_path)