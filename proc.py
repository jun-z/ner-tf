from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import glob
import pickle
import tensorflow as tf

tf.app.flags.DEFINE_string('data_dir', './data', 'Data directory.')
tf.app.flags.DEFINE_integer('max_vocab_size', 1000, 'Max vocabulary size')

FLAGS = tf.app.flags.FLAGS


def init(inputs):
    tokens = {}
    labels = set()
    lengths = []

    if isinstance(inputs, str):
        inputs = [inputs]

    for fn in inputs:
        with open(fn) as f:
            length = 0
            for line in f:
                if line == '\n' and length > 0:
                    lengths.append(length)
                    length = 0
                else:
                    length += 1
                    token, label = line.split()
                    if token in tokens:
                        tokens[token] += 1
                    else:
                        tokens[token] = 1
                    labels.add(label)

    vocab = ['<pad>', '<unk>'] + sorted(tokens, key=tokens.get, reverse=True)

    if len(vocab) > FLAGS.max_vocab_size:
        vocab = vocab[:FLAGS.max_vocab_size]

    labels = ['P'] + sorted(labels)
    return vocab, labels, max(lengths)


def write_list(l, kind):
    with open(os.path.join(FLAGS.data_dir, 'ner.%s' % kind), 'w') as f:
        f.write('\n'.join(l))


def write_data(data, kind):
    with open(os.path.join(FLAGS.data_dir, 'ner.%s.pkl' % kind), 'wb') as f:
        pickle.dump(data, f, 2)


def proc():
    files = {}
    for k in ['train', 'valid', 'test']:
        files[k] = glob.glob(os.path.join(FLAGS.data_dir, '*.%s.iob' % k))

    _vocab, _labels, _length = init(
        files['train'] + files['valid'] + files['test'])

    print('Label size: %i' % len(_labels))
    print('Vocab size: %i' % len(_vocab))
    print('Max sequence length: %i' % _length)

    data = {}
    for k in ['train', 'valid', 'test']:
        data[k] = {'tokens': [], 'labels': [], 'lengths': [], 'weights': []}

    for k, v in files.iteritems():
        for fn in v:
            with open(fn) as f:
                tokens = []
                labels = []
                for line in f:
                    if line == '\n':
                        length = len(tokens)
                        weights = [1] * len(tokens) + [0] * (_length - length)

                        tokens += [0] * (_length - length)
                        labels += [0] * (_length - length)

                        data[k]['tokens'].append(tokens)
                        data[k]['labels'].append(labels)
                        data[k]['lengths'].append(length)
                        data[k]['weights'].append(weights)

                        tokens = []
                        labels = []
                    else:
                        token, label = line.split()
                        if token in _vocab:
                            tokens.append(_vocab.index(token))
                        else:
                            tokens.append(1)

                        labels.append(_labels.index(label))

    write_list(_vocab, 'vocab')
    write_list(_labels, 'labels')

    for k, v in data.iteritems():
        write_data(v, k)


def main(_):
    proc()


if __name__ == '__main__':
    tf.app.run()
