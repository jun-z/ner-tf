from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import glob
import pickle
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('data_dir', './data', 'Data directory.')
tf.app.flags.DEFINE_integer('max_vocab_size', 1000, 'Max vocabulary size.')
tf.app.flags.DEFINE_float('test_split', .2, 'Split for testing data.')
tf.app.flags.DEFINE_float('valid_split', .2, 'Split for validation data.')
tf.app.flags.DEFINE_bool('split_data', False, 'Split dataset.')
tf.app.flags.DEFINE_bool('pretrained_embs', False, 'Use word vectors.')

FLAGS = tf.app.flags.FLAGS


def init(inputs):
    if not FLAGS.pretrained_embs:
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
                    if not FLAGS.pretrained_embs:
                        if token in tokens:
                            tokens[token] += 1
                        else:
                            tokens[token] = 1
                    labels.add(label)

    labels = ['P'] + sorted(labels)

    if FLAGS.pretrained_embs:
        vocab, emb_size = get_vocab()
        return vocab, labels, max(lengths), emb_size
    else:
        vocab = get_vocab(tokens)
        return vocab, labels, max(lengths)


def get_path(fn):
    return os.path.join(FLAGS.data_dir, fn)


def get_vocab(tokens=None):
    specials = ['<pad>', '<unk>']
    if tokens:
        vocab = specials + sorted(tokens, key=tokens.get, reverse=True)
        if len(vocab) > FLAGS.max_vocab_size:
            vocab = vocab[:FLAGS.max_vocab_size]
        return vocab
    else:
        embs = glob.glob(get_path('*.vec'))
        tokens = []

        if len(embs) == 0:
            raise Exception('Did not find file with .vec ext.')
        elif len(embs) > 1:
            raise Exception('Found more than one file with .vec ext.')

        with open(embs[0]) as f:
            for i, line in enumerate(f):
                if i == 0:
                    emb_size = int(line.split()[1])
                else:
                    tokens.append(line.split()[0])
        return specials + tokens, emb_size


def split_data(inputs):
    if isinstance(inputs, str):
        inputs = [inputs]

    if len(inputs) == 0:
        raise Exception('Did not find file with .iob ext.')
    elif len(inputs) > 1:
        raise Exception('Found more than one file with .iob ext.')

    for k in ['train', 'valid', 'test']:
        fn = 'ner.%s.iob' % k
        if os.path.isfile(fn):
            raise Exception('Found file %s.' % fn)

    with open(inputs[0]) as f:
        record = []
        for line in f:
            if line == '\n' and record:
                record.append(line)
                rn = np.random.random()
                if rn < FLAGS.test_split:
                    with open(get_path('ner.test.iob'), 'a') as f:
                        f.writelines(record)
                elif rn < FLAGS.test_split + FLAGS.valid_split:
                    with open(get_path('ner.valid.iob'), 'a') as f:
                        f.writelines(record)
                else:
                    with open(get_path('ner.train.iob'), 'a') as f:
                        f.writelines(record)
                record = []
            else:
                record.append(line)


def write_list(l, kind):
    with open(get_path('ner.%s' % kind), 'w') as f:
        f.write('\n'.join(l))


def write_data(data, kind):
    with open(get_path('ner.%s.pkl' % kind), 'wb') as f:
        pickle.dump(data, f, 2)


def proc():
    if FLAGS.split_data:
        split_data(glob.glob(get_path('*.iob')))

        files = {
            'train': [get_path('ner.train.iob')],
            'valid': [get_path('ner.valid.iob')],
            'test':  [get_path('ner.test.iob')]
        }
    else:
        files = {}
        for k in ['train', 'valid', 'test']:
            files[k] = glob.glob(get_path('*.%s.iob' % k))

    if FLAGS.pretrained_embs:
        _vocab, _labels, _length, emb_size = init(
            files['train'] + files['valid'] + files['test'])
        print('Embedding size: %i' % emb_size)

    else:
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
                    if line == '\n' and tokens and labels:
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

    if FLAGS.pretrained_embs:
        write_list(_vocab, 'vocab')

    write_list(_labels, 'labels')

    for k, v in data.iteritems():
        write_data(v, k)


def main(_):
    proc()


if __name__ == '__main__':
    tf.app.run()
