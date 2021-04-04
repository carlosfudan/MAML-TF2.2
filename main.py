#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21-4-4 上午11:31
# @Author  : carlosliu
# @File : main.py

import os
import time

import tensorflow as tf
from tqdm import tqdm

import config as cfg
from dataReader import *
from meta_learner import MetaLearner
from train import maml_eval, maml_train_on_batch


def load_model(model, dir):
    ckpt = tf.train.Checkpoint(maml_model=model)
    weights = tf.train.latest_checkpoint(dir)
    ckpt.restore(weights)
    return model


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    maml_model = MetaLearner()

    if not cfg.mode:
        maml_model = load_model(maml_model, cfg.ckpt_dir)
        maml_model = MetaLearner.initialize(maml_model)
        maml_model.summary()
        _, test_list = read_omniglot("./datasets/omniglot/images_evaluation")
        test_dataset = task_split(test_list, q_query=cfg.q_query, n_way=cfg.n_way, k_shot=cfg.k_shot)
        test_iter = DataIter(test_dataset)
        test_step = len(test_dataset) // cfg.eval_batch_size
        begin = time.time()
        maml_eval(maml_model, test_iter, test_step, lr_inner=cfg.lr, batch_size=cfg.test_batch_size)
        end = time.time()
        print("\rFinetune time (per task): ", (end - begin) / test_step)
        return

    maml_model = MetaLearner.initialize(maml_model)
    maml_model.summary()
    checkpoint = tf.train.Checkpoint(maml_model=maml_model)

    train_list, valid_list = read_omniglot("./datasets/omniglot/images_background")
    train_dataset = task_split(train_list, q_query=cfg.q_query, n_way=cfg.n_way, k_shot=cfg.k_shot)
    valid_dataset = task_split(valid_list, q_query=cfg.q_query, n_way=cfg.n_way, k_shot=cfg.k_shot)

    train_iter = DataIter(train_dataset)
    valid_iter = DataIter(valid_dataset)

    train_step = len(train_dataset) // cfg.batch_size
    valid_step = len(valid_dataset) // cfg.eval_batch_size

    # 删除上次训练留下的summary文件
    if not os.path.exists(cfg.log_dir):
        os.mkdir(cfg.log_dir)
    for file in os.listdir(cfg.log_dir):
        os.remove(os.path.join(cfg.log_dir, file))

    # 创建summary
    summary_writer = tf.summary.create_file_writer(logdir=cfg.log_dir)

    for epoch in range(1, cfg.epochs + 1):
        train_loss = []
        train_acc = []

        # train
        process_bar = tqdm(range(train_step), ncols=100, desc="Epoch {}".format(epoch), unit="step")
        for _ in process_bar:
            batch_task = get_meta_batch(train_iter, cfg.batch_size)
            loss, acc = maml_train_on_batch(maml_model,
                                            batch_task,
                                            n_way=cfg.n_way,
                                            k_shot=cfg.k_shot,
                                            q_query=cfg.q_query,
                                            lr_inner=cfg.inner_lr,
                                            lr_outer=cfg.outer_lr,
                                            inner_train_step=1)

            train_loss.append(loss)
            train_acc.append(acc)
            process_bar.set_postfix({'loss': '{:.5f}'.format(loss), 'acc': '{:.5f}'.format(acc)})
        
        # 输出平均后的训练结果
        print("\rtrain_loss:{:.4f} train_accuracy:{:.4f}".format(np.mean(train_loss), np.mean(train_acc)))

        if epoch % 10 == 0:
            checkpoint.save(cfg.ckpt_dir+'maml_model.ckpt')
            print("\rCheckpoint is saved")

        val_acc = []
        val_loss = []

        # valid
        process_bar = tqdm(range(valid_step), ncols=100, desc="Epoch {}".format(epoch), unit="step")
        for _ in process_bar:
            batch_task = get_meta_batch(valid_iter, cfg.eval_batch_size)
            loss, acc = maml_train_on_batch(maml_model,
                                            batch_task,
                                            n_way=cfg.n_way,
                                            k_shot=cfg.k_shot,
                                            q_query=cfg.q_query,
                                            lr_inner=cfg.inner_lr,
                                            lr_outer=cfg.outer_lr,
                                            inner_train_step=3,
                                            meta_update=False)
            val_loss.append(loss)
            val_acc.append(acc)

            process_bar.set_postfix({'val_loss': '{:.5f}'.format(loss), 'val_acc': '{:.5f}'.format(acc)})

        # 输出平均后的验证结果
        print("\rvalidation_loss:{:.4f} validation_accuracy:{:.4f}\n".format(np.mean(val_loss), np.mean(val_acc)))

        # 保存到tensorboard里
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', np.mean(train_loss), step=epoch)
            tf.summary.scalar('train_acc', np.mean(train_acc), step=epoch)
            tf.summary.scalar('valid_loss', np.mean(val_loss), step=epoch)
            tf.summary.scalar('valid_acc', np.mean(val_acc), step=epoch)

    # maml_model.save_weights(cfg.save_path)


if __name__ == '__main__':
    main()
