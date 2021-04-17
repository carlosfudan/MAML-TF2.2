#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21-4-17 下午12:57
# @Author  : carlosliu
# @File    : svm.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def svm():
    return Pipeline(( ("scaler", StandardScaler()), ("linear_svc", SVC(kernel="rbf", gamma=0.1, C=0.001))))
