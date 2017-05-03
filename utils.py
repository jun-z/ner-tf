from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cPickle as pickle


def load_data(data_dir):
    files = os.listdir(data_dir)
    for f in files:
        if f.endswith('.train.pkl'):
            with open(os.path.join(data_dir, f), 'rb') as _f:
                train_set = pickle.load(_f)
        if f.endswith('.valid.pkl'):
            with open(os.path.join(data_dir, f), 'rb') as _f:
                valid_set = pickle.load(_f)
        if f.endswith('.test.pkl'):
            with open(os.path.join(data_dir, f), 'rb') as _f:
                test_set = pickle.load(_f)
    return train_set, valid_set, test_set


def load_list(data_dir, ext):
    files = os.listdir(data_dir)
    for f in files:
        if f.endswith(ext):
            with open(os.path.join(data_dir, f)) as f:
                lines = f.readlines()
    vocab = [line[:-1] if line.endswith('\n') else line for line in lines]
    return vocab


def accuracy(preds, labels, length):
    label_c = label_t = 0
    record_c = record_t = len(labels)
    for i in range(record_t):
        record_a = True
        for j in range(length[i]):
            if preds[i][j] == labels[i][j]:
                label_c += 1
            else:
                record_a = False
            label_t += 1
        if not record_a:
            record_c -= 1
    return label_c / label_t, record_c / record_t
