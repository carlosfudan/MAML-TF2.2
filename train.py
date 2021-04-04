#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21-4-4 上午11:31
# @Author  : carlosliu
# @File : train.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers
from tqdm import tqdm

import config as cfg
from dataReader import create_label, get_meta_batch
from meta_learner import MetaLearner


def copy_model(model, x):
    copied_model = MetaLearner()
    copied_model(x)
    copied_model.set_weights(model.get_weights())
    return copied_model


def compute_loss(y_true, y_pred):
    """
    计算loss
    :param y_true: 模型
    :param y_pred:
    :return:
    """
    mse = losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

    return mse


def maml_eval(model,
              test_iter,
              test_step,
              n_way=5,
              k_shot=1,
              q_query=1,
              lr_inner=0.001,
              batch_size=2):
    """
    maml的测试函数
    :param model: 经过训练后的模型
    :param test_iter: 测试集的迭代器对象
    :param n_way: 一个任务内分类的数量
    :param k_shot: support set
    :param q_query: query set
    :param lr_inner: 内层support set的学习率
    :param batch_size:
    :return: None
    """

    test_acc = []
    test_loss = []

    for batch_id in range(test_step):
        batch_task = get_meta_batch(test_iter, batch_size)
        loss, acc = maml_val_on_batch(model,
                                        batch_task,
                                        n_way=n_way,
                                        k_shot=k_shot,
                                        q_query=q_query,
                                        lr_inner=lr_inner,
                                        inner_train_step=cfg.finetune_step)
        test_acc.append(loss)
        test_loss.append(acc)

        # 输出测试结果
        print("test_loss:{:.4f} test_accuracy:{:.4f}".format(np.mean(test_acc), np.mean(test_loss)))

def maml_train_on_batch(model,
                        batch_task,
                        n_way=5,
                        k_shot=1,
                        q_query=1,
                        lr_inner=0.1,
                        lr_outer=0.0002,
                        inner_train_step=1,
                        meta_update=True):
    """
    根据论文上Algorithm 1上的流程进行模型的训练
    :param model: MAML的模型
    :param batch_task: 一个batch 的任务
    :param n_way: 一个任务内分类数量
    :param k_shot: support set的数量
    :param q_query: query的数量
    :param lr_inner: 内层support set的学习率
    :param lr_outer: 外层query set任务的学习率
    :param inner_train_step: 内层support set的训练次数
    :param meta_update: 是否进行meta update
    :return: loss, accuracy -- 都是均值
    """
    outer_optimizer = optimizers.Adam(lr_outer)

    task_loss = []
    task_acc = []


    with tf.GradientTape() as query_tape:
        for one_task in batch_task:

            support_x = one_task[:n_way * k_shot]
            query_x = one_task[n_way * k_shot:]
            support_y = create_label(n_way, k_shot)
            query_y = create_label(n_way, q_query)

            if meta_update:
                copied_model = model
            else:
                copied_model = copy_model(model, support_x)

            for inner_step in range(inner_train_step):
                with tf.GradientTape(watch_accessed_variables=False) as inner_tape:
                    inner_tape.watch(copied_model.inner_weights)
                    support_logits, _ = copied_model(support_x)
                    support_loss = compute_loss(support_y, support_logits)

                inner_grads = inner_tape.gradient(support_loss, copied_model.inner_weights)
                copied_model = MetaLearner.meta_update(copied_model, alpha=lr_inner, grads=inner_grads)

            query_logits, query_pred = copied_model(query_x)
            query_loss = compute_loss(query_y, query_logits)

            equal_list = tf.equal(tf.argmax(query_pred, -1), tf.cast(query_y, tf.int64))
            acc = tf.reduce_mean(tf.cast(equal_list, tf.float32))
            task_acc.append(acc)
            task_loss.append(query_loss)

        meta_batch_loss = tf.reduce_mean(task_loss)

    if not meta_update:
        del copied_model

    if meta_update:
        outer_grads = query_tape.gradient(meta_batch_loss, model.trainable_variables)
        outer_optimizer.apply_gradients(zip(outer_grads, model.trainable_variables))
    return meta_batch_loss, np.mean(task_acc)

def maml_val_on_batch(model,
                        batch_task,
                        n_way=5,
                        k_shot=1,
                        q_query=1,
                        lr_inner=0.1,
                        inner_train_step=1):
    optimizer = optimizers.SGD(lr_inner)

    task_loss = []
    task_acc = []


    with tf.GradientTape() as query_tape:
        for one_task in batch_task:

            support_x = one_task[:n_way * k_shot]
            query_x = one_task[n_way * k_shot:]
            support_y = create_label(n_way, k_shot)
            query_y = create_label(n_way, q_query)

            copied_model = model

            for inner_step in range(inner_train_step):
                with tf.GradientTape() as tape:
                    tape.watch(copied_model.inner_weights)
                    support_logits, _ = copied_model(support_x)
                    support_loss = compute_loss(support_y, support_logits)

                grads = tape.gradient(support_loss, copied_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, copied_model.trainable_variables))

                query_logits, query_pred = copied_model(query_x)
                query_loss = compute_loss(query_y, query_logits)

                equal_list = tf.equal(tf.argmax(query_pred, -1), tf.cast(query_y, tf.int64))
                acc = tf.reduce_mean(tf.cast(equal_list, tf.float32))
                task_acc.append(acc)
                task_loss.append(query_loss)

        meta_batch_loss = tf.reduce_mean(tf.stack(task_loss))

    return meta_batch_loss, np.mean(task_acc)

