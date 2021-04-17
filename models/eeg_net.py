#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21-4-14 下午9:09
# @Author  : carlosliu
# @File    : eeg_net.py

from tensorflow.keras import Model, activations, layers, models, optimizers

import config as cfg


def cnn(num_classes=2, width=cfg.eeg_width, height=cfg.eeg_height, channel=cfg.eeg_channel, time_step=20):
    net = models.Sequential([
        layers.Reshape((width, height, time_step)),

        layers.BatchNormalization(),

        layers.Conv2D(filters=8, kernel_size=3, strides=(1, 1), padding="SAME",
                                             activation="relu", input_shape=[(None, width, height, channel)]),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(num_classes),
        layers.Softmax()
    ])
    inputs = layers.Input(shape=(time_step, width, height, channel))
    output = net(inputs)
    model = Model(inputs=inputs, outputs=output)

    return model


def eeg_net(num_classes=2, width=cfg.eeg_width, height=cfg.eeg_height, channel=cfg.eeg_channel, time_step=20):
    eeg_net = models.Sequential([
        layers.Reshape((time_step, width, height, channel)),

        layers.BatchNormalization(),

        layers.TimeDistributed(layers.Conv2D(filters=32, kernel_size=1, strides=(1,1), padding="SAME",
                                             activation="relu", input_shape=[(None,width,height,channel)])),
        layers.BatchNormalization(),

        layers.TimeDistributed(layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding="SAME",
                                             activation="elu")),
        layers.BatchNormalization(),

        layers.TimeDistributed(layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding="SAME",
                                             activation="elu")),
        layers.BatchNormalization(),

        layers.TimeDistributed(layers.Flatten()),
        layers.TimeDistributed(layers.Dense(64, activation="relu")),
        layers.Dropout(0.3),

        layers.LSTM(8, return_sequences=True),
        layers.LSTM(8),
        layers.Dense(16, activation="relu"),
        layers.Dense(num_classes),
        layers.Softmax()
    ])
    inputs = layers.Input(shape=(time_step, width, height, channel))
    output = eeg_net(inputs)
    model = Model(inputs=inputs, outputs=output)

    return model

# TODO MAML
# class EEGNet(tf.keras.models.Model):
#     def __init__(self):
#         super(EEGNet, self).__init__()
#         self.num_classes = 2
#         self.width = cfg.eeg_width
#         self.height = cfg.eeg_height
#         self.channel=cfg.eeg_channel
#         self.time_step = 20
#         self.training = True
#         self.ip_size = (self.time_step, self.width, self.height, self.channel)
#
#         self.reshape = tf.keras.layers.Reshape((self.time_step, self.width, self.height, -1))
#
#         self.conv_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),
#                                                                              padding='SAME', kernel_initializer='glorot_normal'))
#         self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1)
#
#         self.conv_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
#                                                                              padding='SAME', kernel_initializer='glorot_normal'))
#         self.bn_2 = tf.keras.layers.BatchNormalization(axis=-1)
#
#         self.conv_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
#                                                                              padding='SAME', kernel_initializer='glorot_normal'))
#         self.bn_3 = tf.keras.layers.BatchNormalization(axis=-1)
#
#         self.flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
#
#         self.fc_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256))
#         self.dropdout = tf.keras.layers.Dropout(0.5)
#
#         self.lstm_1 = tf.keras.layers.LSTM(8, return_sequences=True)
#         self.lstm_2 = tf.keras.layers.LSTM(8)
#         self.fc_2 = tf.keras.layers.Dense(256, activation="elu")
#         self.out = tf.keras.layers.Dense(self.num_classes)
#
#     @classmethod
#     def initialize(cls, model):
#         ip_size = model.ip_size
#         model.build(ip_size)
#         return model
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.reshape(inputs)
#         x = tf.keras.activations.elu(self.bn_1(self.conv_1(x), training=self.training))
#         x = tf.keras.activations.elu(self.bn_2(self.conv_2(x), training=self.training))
#         x = tf.keras.activations.elu(self.bn_3(self.conv_3(x), training=self.training))
#
#         x = self.flatten(x)
#         x = self.fc_1(x, training=self.training)
#         x = self.dropdout(x, training=self.training)
#
#         x = self.lstm_1(x, training=self.training)
#         x = self.lstm_2(x, training=self.training)
#
#         x = self.fc_2(x, training=self.training)
#
#         logits = self.out(x, training=self.training)
#         pred = tf.keras.activations.softmax(logits)
#
#         return pred