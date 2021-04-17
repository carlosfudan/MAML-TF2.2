#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21-4-4 上午11:31
# @Author  : carlosliu
# @File : config.py

# rain model -> please set cfg.mode = True, test model -> please set cfg.mode = False
mode = True

batch_size = 32
eval_batch_size = 4
test_batch_size = 4


epochs = 10
with_bn = True

lr = 0.0005
inner_lr = 0.005
outer_lr = 1e-3
finetune_step = 30

n_way = 5
k_shot = 1
q_query = 1

width = 32
height = 32
channel = 1

eeg_width = 8
eeg_height = 11
eeg_channel = 1

ckpt_dir = "./logs/ckpt/"
save_path = "./logs/model/maml.h5"
log_dir = "./logs/summary/"

