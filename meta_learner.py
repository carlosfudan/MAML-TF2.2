#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21-4-4 上午11:31
# @Author  : carlosliu
# @File    : meta_learner.py

import tensorflow as tf

import config as cfg


def loss_fn(y, pred_y):
    return tf.reduce_mean(tf.losses.categorical_crossentropy(y, pred_y))


class MetaLearner(tf.keras.models.Model):
    def __init__(self, bn=None):
        super(MetaLearner, self).__init__()
        self.filters = 64
        self.ip_size = (1, cfg.width, cfg.height, 1)
        self.op_channel = cfg.n_way
        self.with_bn = cfg.with_bn
        self.training = cfg.mode

        if self.with_bn is True:
            self.conv_1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3), strides=(1,1), padding='SAME', kernel_initializer='glorot_normal')
            self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1)
            self.max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

            self.conv_2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3), strides=(1,1), padding='SAME', kernel_initializer='glorot_normal')
            self.bn_2 = tf.keras.layers.BatchNormalization(axis=-1)
            self.max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

            self.conv_3 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3), strides=(1,1), padding='SAME', kernel_initializer='glorot_normal')
            self.bn_3 = tf.keras.layers.BatchNormalization(axis=-1)
            self.max_pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

            self.conv_4 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3), strides=(1,1), padding='SAME', kernel_initializer='glorot_normal')
            self.bn_4 = tf.keras.layers.BatchNormalization(axis=-1)
            self.max_pool_4 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

            self.fc = tf.keras.layers.Flatten()
            self.out = tf.keras.layers.Dense(self.op_channel)
        
        elif self.with_bn is False:
            self.conv_1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3), strides=(1,1), padding='SAME', kernel_initializer='glorot_normal')
            self.max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

            self.conv_2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3), strides=(1,1), padding='SAME', kernel_initializer='glorot_normal')
            self.max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

            self.conv_3 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3), strides=(1,1), padding='SAME', kernel_initializer='glorot_normal')
            self.max_pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

            self.conv_4 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3), strides=(1,1), padding='SAME', kernel_initializer='glorot_normal')
            self.max_pool_4 = tf.keras.layers.MaxPool2D(pool_size=(2,2))

            self.fc = tf.keras.layers.Flatten()
            self.out = tf.keras.layers.Dense(self.op_channel)
    
    @property
    def inner_weights(self):
        if self.with_bn is True:
            weights = [
                self.conv_1.kernel, self.conv_1.bias,
                self.bn_1.gamma, self.bn_1.beta,
                self.conv_2.kernel, self.conv_2.bias,
                self.bn_2.gamma, self.bn_2.beta,
                self.conv_3.kernel, self.conv_3.bias,
                self.bn_3.gamma, self.bn_3.beta,
                self.conv_4.kernel, self.conv_4.bias,
                self.bn_4.gamma, self.bn_4.beta,
                self.out.kernel, self.out.bias
            ]   
        elif self.with_bn is False:
            weights = [
                self.conv_1.kernel, self.conv_1.bias,
                self.conv_2.kernel, self.conv_2.bias,
                self.conv_3.kernel, self.conv_3.bias,
                self.conv_4.kernel, self.conv_4.bias,
                self.out.kernel, self.out.bias
            ]
        return weights

    @classmethod
    def initialize(cls, model):
        ip_size = model.ip_size
        model.build(ip_size)
        return model

    @classmethod
    def hard_copy(cls, model):
        copied_model = cls()
        copied_model.build(model.ip_size)

        if copied_model.with_bn is True:
            copied_model.conv_1.kernel = model.conv_1.kernel
            copied_model.conv_1.bias = model.conv_1.bias
            copied_model.bn_1.gamma = model.bn_1.gamma
            copied_model.bn_1.beta = model.bn_1.beta
            # copied_model.max_pool_1 = model.max_pool_1

            copied_model.conv_2.kernel = model.conv_2.kernel
            copied_model.conv_2.bias = model.conv_2.bias
            copied_model.bn_2.gamma = model.bn_2.gamma
            copied_model.bn_2.beta = model.bn_2.beta
            # copied_model.max_pool_2 = model.max_pool_2
            
            copied_model.conv_3.kernel = model.conv_3.kernel
            copied_model.conv_3.bias = model.conv_3.bias
            copied_model.bn_3.gamma = model.bn_3.gamma
            copied_model.bn_3.beta = model.bn_3.beta
            # copied_model.max_pool_3 = model.max_pool_3

            copied_model.conv_4.kernel = model.conv_4.kernel
            copied_model.conv_4.bias = model.conv_4.bias
            copied_model.bn_4.gamma = model.bn_4.gamma
            copied_model.bn_4.beta = model.bn_4.beta
            # copied_model.max_pool_4 = model.max_pool_4

            copied_model.out.kernel = model.out.kernel
            copied_model.out.bias = model.out.bias
            
        elif copied_model.with_bn is False:
            copied_model.conv_1.kernel = model.conv_1.kernel
            copied_model.conv_1.bias = model.conv_1.bias
            # copied_model.max_pool_1 = model.max_pool_1

            copied_model.conv_2.kernel = model.conv_2.kernel
            copied_model.conv_2.bias = model.conv_2.bias
            # copied_model.max_pool_2 = model.max_pool_2
            
            copied_model.conv_3.kernel = model.conv_3.kernel
            copied_model.conv_3.bias = model.conv_3.bias
            # copied_model.max_pool_3 = model.max_pool_3

            copied_model.conv_4.kernel = model.conv_4.kernel
            copied_model.conv_4.bias = model.conv_4.bias
            # copied_model.max_pool_4 = model.max_pool_4

            copied_model.out.kernel = model.out.kernel
            copied_model.out.bias = model.out.bias
        
        return copied_model

    
    @classmethod
    def meta_update(cls, model, alpha=0.01, grads=None):
        copied_model = cls()
        copied_model.build(model.ip_size)

        if copied_model.with_bn is True:
            copied_model.conv_1.kernel = model.conv_1.kernel
            copied_model.conv_1.bias = model.conv_1.bias
            copied_model.bn_1.gamma = model.bn_1.gamma
            copied_model.bn_1.beta = model.bn_1.beta
            # copied_model.max_pool_1 = model.max_pool_1

            copied_model.conv_2.kernel = model.conv_2.kernel
            copied_model.conv_2.bias = model.conv_2.bias
            copied_model.bn_2.gamma = model.bn_2.gamma
            copied_model.bn_2.beta = model.bn_2.beta
            # copied_model.max_pool_2 = model.max_pool_2
            
            copied_model.conv_3.kernel = model.conv_3.kernel
            copied_model.conv_3.bias = model.conv_3.bias
            copied_model.bn_3.gamma = model.bn_3.gamma
            copied_model.bn_3.beta = model.bn_3.beta
            # copied_model.max_pool_3 = model.max_pool_3

            copied_model.conv_4.kernel = model.conv_4.kernel
            copied_model.conv_4.bias = model.conv_4.bias
            copied_model.bn_4.gamma = model.bn_4.gamma
            copied_model.bn_4.beta = model.bn_4.beta
            # copied_model.max_pool_4 = model.max_pool_4

            copied_model.out.kernel = model.out.kernel
            copied_model.out.bias = model.out.bias

            if grads is not None:
                # compute SGD
                copied_model.conv_1.kernel = copied_model.conv_1.kernel - alpha * grads[0]
                copied_model.conv_1.bias = copied_model.conv_1.bias - alpha * grads[1]
                copied_model.bn_1.gamma = copied_model.bn_1.gamma - alpha * grads[2]
                copied_model.bn_1.beta = copied_model.bn_1.beta - alpha * grads[3]

                copied_model.conv_2.kernel = copied_model.conv_2.kernel - alpha * grads[4]
                copied_model.conv_2.bias = copied_model.conv_2.bias - alpha * grads[5]
                copied_model.bn_2.gamma = copied_model.bn_2.gamma - alpha * grads[6]
                copied_model.bn_2.beta = copied_model.bn_2.beta - alpha * grads[7]

                copied_model.conv_3.kernel = copied_model.conv_3.kernel - alpha * grads[8]
                copied_model.conv_3.bias = copied_model.conv_3.bias - alpha * grads[9]
                copied_model.bn_3.gamma = copied_model.bn_3.gamma - alpha * grads[10]
                copied_model.bn_3.beta = copied_model.bn_3.beta - alpha * grads[11]

                copied_model.conv_4.kernel = copied_model.conv_4.kernel - alpha * grads[12]
                copied_model.conv_4.bias = copied_model.conv_4.bias - alpha * grads[13]
                copied_model.bn_4.gamma = copied_model.bn_4.gamma - alpha * grads[14]
                copied_model.bn_4.beta = copied_model.bn_4.beta - alpha * grads[15]

                copied_model.out.kernel = copied_model.out.kernel - alpha * grads[16]
                copied_model.out.bias = copied_model.out.bias - alpha * grads[17]

        elif copied_model.with_bn is False:
            copied_model.conv_1.kernel = model.conv_1.kernel
            copied_model.conv_1.bias = model.conv_1.bias
            # copied_model.max_pool_1 = model.max_pool_1

            copied_model.conv_2.kernel = model.conv_2.kernel
            copied_model.conv_2.bias = model.conv_2.bias
            # copied_model.max_pool_2 = model.max_pool_2
            
            copied_model.conv_3.kernel = model.conv_3.kernel
            copied_model.conv_3.bias = model.conv_3.bias
            # copied_model.max_pool_3 = model.max_pool_3

            copied_model.conv_4.kernel = model.conv_4.kernel
            copied_model.conv_4.bias = model.conv_4.bias
            # copied_model.max_pool_4 = model.max_pool_4

            copied_model.out.kernel = model.out.kernel
            copied_model.out.bias = model.out.bias

            if grads is not None:
                copied_model.conv_1.kernel = copied_model.conv_1.kernel - alpha * grads[0]
                copied_model.conv_1.bias = copied_model.conv_1.bias - alpha * grads[1]

                copied_model.conv_2.kernel = copied_model.conv_2.kernel - alpha * grads[2]
                copied_model.conv_2.bias = copied_model.conv_2.bias - alpha * grads[3]

                copied_model.conv_3.kernel = copied_model.conv_3.kernel - alpha * grads[4]
                copied_model.conv_3.bias = copied_model.conv_3.bias - alpha * grads[5]

                copied_model.conv_4.kernel = copied_model.conv_4.kernel - alpha * grads[6]
                copied_model.conv_4.bias = copied_model.conv_4.bias - alpha * grads[7]

                copied_model.out.kernel = copied_model.out.kernel - alpha * grads[8]
                copied_model.out.bias = copied_model.out.bias - alpha * grads[9]
        
        return copied_model

    def call(self, x):
        if self.with_bn is True:
            x = self.max_pool_1(tf.keras.activations.relu(self.bn_1(self.conv_1(x), training=self.training)))
            x = self.max_pool_2(tf.keras.activations.relu(self.bn_2(self.conv_2(x), training=self.training)))
            x = self.max_pool_3(tf.keras.activations.relu(self.bn_3(self.conv_3(x), training=self.training)))
            x = self.max_pool_4(tf.keras.activations.relu(self.bn_4(self.conv_4(x), training=self.training)))
        
        elif self.with_bn is False:
            x = self.max_pool_1(tf.keras.activations.relu(self.conv_1(x)))
            x = self.max_pool_2(tf.keras.activations.relu(self.conv_2(x)))
            x = self.max_pool_3(tf.keras.activations.relu(self.conv_3(x)))
            x = self.max_pool_4(tf.keras.activations.relu(self.conv_4(x)))

        x = self.fc(x)
        logits = self.out(x)
        pred = tf.keras.activations.softmax(logits)
        
        return logits, pred

