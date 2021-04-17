#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21-4-4 上午11:31
# @Author  : carlosliu
# @File : dataReader.py

import copy
import os
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf

import config as cfg


class DataIter(object):
    """
    将数据集路径列表变成迭代器，加快数据读取速度
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.current_num = 0

    # 使用iter()方法的时候,会调用__iter__
    def __iter__(self):
        # 简化之后,这里就要返回可迭代对象的本身
        return self

    def __next__(self):
        # 防止迭代的列表越界
        if self.current_num >= len(self.dataset):
            # 当我们把列表的元素取完之后,要产生停止信号,
            # 但在这里为了能够不停的取数据，可以把current_num置为0，以达到tf.data的repeat的效果
            self.current_num = 0

        ret = self.dataset[self.current_num]
        self.current_num += 1

        return ret


def read_eeg_dataset(path):
    sample_path = path + "3+3wyh_ntntwX2c.pkl"
    label_path = path + "3+3wyh_ntntwy2c.pkl"
    with open(sample_path, "rb") as f:
        X = pickle.load(f)
    with open(label_path, "rb") as f:
        Y = pickle.load(f)

    sample_path1 = path + "3+3wyh_ntntwfbX2c.pkl"
    label_path1 = path + "3+3wyh_ntntwfby2c.pkl"
    with open(sample_path1, "rb") as f:
        X1 = pickle.load(f)
    with open(label_path1, "rb") as f:
        Y1 = pickle.load(f)

    X = np.concatenate((X, X1), 0)
    Y = np.concatenate((Y, Y1), 0)

    from sklearn.model_selection import train_test_split
    return train_test_split(X, Y, test_size=0.2, shuffle=True)


def read_omniglot(path):
    """
    读取omniglot，将其存入列表种
    :param path:
    :return:
    """
    classes = []

    for alphabet in os.listdir(path):
        # 语言路径
        alphabet_path = os.path.join(path, alphabet)

        for letter in os.listdir(alphabet_path):
            # 字体路径
            letter_path = os.path.join(alphabet_path, letter)
            letter_class = []
            # 具体图片路径
            for img_name in os.listdir(letter_path):
                img_path = os.path.join(letter_path, img_name)
                letter_class.append(os.path.normpath(img_path))
            classes.append(letter_class)

    rate = int(len(classes) * 0.8)
    train, valid = classes[:rate], classes[rate:]

    return train, valid


def read_enhance_omniglot(path):
    """
    将各个分类下的img_path，按任务为单位分类。这个API是基于Mini-ImageNet下实现的，其中每个类只有600个
    为了均匀利用到所有数据，(q_query + k_shot) * n_way 要能被 图片数量整除
    。n-way * k-shot张图片用来给inner loop训练，n-way * query是给out loop去test
    dataset最终是 , shape = [batch_size, n_way * (k_shot + q_query), 1, 28, 28]
    :return:
    """
    classes = [[] for _ in range(964)]

    # 存储不同字符的索引 data augmentation之后的数据和data是同索引
    index = dict()

    # classes的索引
    i = -1

    for alphabet in os.listdir(path):
        alphabet_path = os.path.join(path, alphabet)

        for letter in os.listdir(alphabet_path):
            letter_path = os.path.join(alphabet_path, letter)
            key = "{}-{}".format(alphabet.split('.')[0], letter)

            if key not in index:
                index.update({key: i})
            else:
                i = index[key]
            i += 1

            # 具体图片路径
            for img_name in os.listdir(letter_path):
                img_path = os.path.join(letter_path, img_name)
                classes[i].append(os.path.normpath(img_path))

    rate = int(len(classes) * 0.8)
    train, valid = classes[:rate], classes[rate:]

    return train, valid


def read_miniimagenet(csv_path, one_class_img=600):
    """
    读取包含图片名和标签的csv
    :param csv_path:
    :param one_class_img: 一个类中有几张图片
    :return:
    """
    csv = pd.read_csv(csv_path)

    image_list = list("./data/miniImageNet/images/" + csv.iloc[:, 0])

    num_class = len(image_list) // one_class_img  # 总共有几类
    classes = [[] for _ in range(num_class)]

    # 先按照类区分开
    for i in range(num_class):
        start = i * one_class_img
        end = (i + 1) * one_class_img
        classes[i] = image_list[start: end]

    return classes


def get_meta_batch(iterator, meta_batch_size):
    """
    生成一个batch的任务，用于训练。将传入的列表中的数据组合成一个batch_size
    :param iterator: 数据集的迭代器对象
    :param meta_batch_size: batch_size个任务组成一个meta_batch
    :return: 生成一个batch的任务
    """
    batch_task = list()

    for _ in range(meta_batch_size):
        one_task_img_path = next(iterator)
        one_task_img_data = process_one_task(one_task_img_path)
        one_task_img_data = tf.squeeze(one_task_img_data, axis=1)
        batch_task.append(one_task_img_data)

    return batch_task


def process_one_task(one_task, width=cfg.width, height=cfg.height):
    """
    对一个任务处理，对其中每一个图片进行读取
    :param one_task: 一个batch的任务[img_path]
    :param width:
    :param height:
    :return:
    """
    task = []

    for img_path in one_task:
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image)
        # 将unit8转为float32且归一化
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [width, height])

        task.append([image])

    return task


def create_label(n_way, k_shot):
    """
    创建标签，生成一个0 - n_way的序列，每个元素重复k_shot次
    :param n_way:
    :param k_shot:
    :return:
    """
    return tf.convert_to_tensor(np.repeat(range(n_way), k_shot), dtype=tf.float32)


def task_split(classes: list, q_query=1, n_way=5, k_shot=1):
    """
    将各个分类下的img_path，按任务为单位分类。这个API是基于Mini-ImageNet下实现的，其中每个类只有600个
    为了均匀利用到所有数据，(q_query + k_shot) * n_way 要能被 图片数量整除
    。n-way * k-shot张图片用来给inner loop训练，n-way * query是给out loop去test
    dataset最终是 , shape = [batch_size, n_way * (k_shot + q_query), 1, 28, 28]
    :param classes: shape为(class_num, img_num)的二位列表，存储了图片的路径
    :param q_query: query-set的数量
    :param n_way: 一个任务由几个类组成
    :param k_shot: support-set数量
    :return:
    """
    dataset = []
    classes = copy.deepcopy(classes)
    # 这样计算循环次数的前提得是每个分类中图片数量相同，下面划分数据集的操作也是基于这个前提才能计算的
    # 总的循环数 = 图片总数 // 一个任务所包含图片的数量
    loop_num = len(classes) * len(classes[0]) // ((q_query + k_shot) * n_way)

    choose = [i for i in range(len(classes))]
    random.shuffle(choose)

    end = 0
    # drop_last来控制丢弃剩余几个元素
    drop_last = False

    for _ in range(loop_num):
        # 用来存储一个任务的图片
        one_task = []
        # 索引有可能会大于列表长度，故需要截断处理，且每次的start都应该是上一个end+1(切片取不到end)
        start = end
        end = (start + n_way) % len(classes)

        if end < start:
            task_class = choose[start:] + choose[:end]
        else:
            task_class = choose[start: end]

        # 循环n_way次，取出k_shot个训练图像
        for i in task_class:
            for _ in range(k_shot):
                if len(classes[i]) <= 0:
                    drop_last = True
                    break

                one_task.append(classes[i].pop(0))

        # 取出q_query个训练图像
        for i in task_class:
            for _ in range(q_query):
                if len(classes[i]) <= 0:
                    drop_last = True
                    break

                one_task.append(classes[i].pop(0))

        if drop_last:
            break

        dataset.append(one_task)

    return dataset
