from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import cPickle as pickle
import tensorflow as tf

from blstm import BLSTM
from utils import load_list

tf.app.flags.DEFINE_string('input', '', 'Input file.')
tf.app.flags.DEFINE_string('output', '', 'Output file.')
tf.app.flags.DEFINE_string('format', 'pkl', 'Input file format.')
tf.app.flags.DEFINE_string('data_dir', './data', 'Data directory.')
tf.app.flags.DEFINE_string('train_dir', './model', 'Training directory.')
tf.app.flags.DEFINE_integer('batch_size', -1, 'Batch size.')
tf.app.flags.DEFINE_integer('num_units', 50, 'Number of units in LSTM.')
tf.app.flags.DEFINE_integer('num_layers', 1, 'Number of LSTM layers.')
tf.app.flags.DEFINE_integer('num_steps', 25, 'Max number of time steps')
tf.app.flags.DEFINE_integer('num_labels', 8, 'Number of labels.')
tf.app.flags.DEFINE_integer('emb_size', 10, 'Size of embedding.')
tf.app.flags.DEFINE_integer('vocab_size', 55, 'Size of vocabulary.')
tf.app.flags.DEFINE_float('learning_rate', .03, 'Learning rate.')
tf.app.flags.DEFINE_float('max_clip_norm', 5.0, 'Clip norm for gradients.')
tf.app.flags.DEFINE_bool('use_crf', False, 'Use CRF loss.')
tf.app.flags.DEFINE_bool('use_fp16', False, 'Use tf.float16.')
tf.app.flags.DEFINE_bool('diff_only', False, 'Output only differences.')
tf.app.flags.DEFINE_bool('true_labels', False, 'If the data has true labels.')

FLAGS = tf.app.flags.FLAGS


def create_model(session):
    model = BLSTM(
        FLAGS.num_units,
        FLAGS.num_layers,
        FLAGS.num_steps,
        FLAGS.num_labels,
        FLAGS.emb_size,
        FLAGS.vocab_size,
        FLAGS.learning_rate,
        FLAGS.max_clip_norm,
        FLAGS.use_crf,
        tf.float16 if FLAGS.use_fp16 else tf.float32)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Restoring model from %s.' % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        return model
    else:
        raise Exception('Could not find model in %s'
                        % ckpt.model_checkpoint_path)


def write_record(f, tokens, labels, true_labels=None):
    if true_labels is None:
        for i, t in zip(tokens, labels):
            f.write(' '.join([i, t]) + '\n')
    else:
        for i, t, p in zip(tokens, labels, true_labels):
            f.write(' '.join([i, t, p]) + '\n')
    f.write('\n')


def read_data():
    _vocab = load_list(FLAGS.data_dir, '.vocab')
    _labels = load_list(FLAGS.data_dir, '.labels')

    if FLAGS.format == 'pkl':
        with open(FLAGS.input, 'rb') as f:
            data = pickle.load(f)
    elif FLAGS.format == 'iob':
        with open(FLAGS.input) as f:
            length = 0
            _length = 0
            for line in f:
                if line == '\n' and length > 0:
                    _length = max(_length, length)
                    length = 0
                else:
                    length += 1
        if _length > FLAGS.num_steps:
            raise ValueError(
                'Max sequence length %i > num_steps %i'
                % (_length, FLAGS.num_steps))
        else:
            _length = FLAGS.num_steps

        with open(FLAGS.input) as f:
            data = {'tokens': [], 'labels': [], 'lengths': [], 'weights': []}
            tokens = []
            labels = []
            for line in f:
                if line == '\n' and tokens and labels:
                    length = len(tokens)
                    weights = [1] * len(tokens) + [0] * (_length - length)

                    tokens += [0] * (_length - length)
                    labels += [0] * (_length - length)

                    data['tokens'].append(tokens)
                    data['labels'].append(labels)
                    data['lengths'].append(length)
                    data['weights'].append(weights)

                    tokens = []
                    labels = []
                else:
                    token, label = line.split()
                    if token in _vocab:
                        tokens.append(_vocab.index(token))
                    else:
                        tokens.append(1)

                    labels.append(_labels.index(label))
    elif FLAGS.format == 'txt':
        raise ValueError('Format not supported yet.')
    else:
        raise ValueError('Unknown file format %s.' % FLAGS.format)
    return _vocab, _labels, data


def eval():
    _vocab, _labels, data = read_data()

    with tf.Session() as sess:
        model = create_model(sess)
        if FLAGS.use_crf:
            logits, trans_params = sess.run(
                [model.logits, model.trans_params], feed_dict={
                    model.tokens: data['tokens'],
                    model.lengths: data['lengths']
                })
            preds = [
                tf.contrib.crf.viterbi_decode(logits[l], trans_params)[0]
                for l in range(logits.shape[0])]
        else:
            probs = sess.run(
                model.probs, feed_dict={
                    model.tokens: data['tokens'],
                    model.lengths: data['lengths']
                })
            preds = np.argmax(probs, axis=2)
        with open(FLAGS.output, 'w') as f:
            for i in range(len(preds)):
                tokens = []
                labels = []
                if FLAGS.true_labels:
                    correct = True
                    true_labels = []
                for j in range(data['lengths'][i]):
                    tokens.append(_vocab[data['tokens'][i][j]])
                    labels.append(_labels[preds[i][j]])
                    if FLAGS.true_labels:
                        true_labels.append(_labels[data['labels'][i][j]])
                        if preds[i][j] != data['labels'][i][j]:
                            correct = False
                if FLAGS.true_labels:
                    if correct:
                        if not FLAGS.diff_only:
                            write_record(f, tokens, labels, true_labels)
                    else:
                        write_record(f, tokens, labels, true_labels)
                else:
                    write_record(f, tokens, labels)


def main(_):
    eval()


if __name__ == '__main__':
    tf.app.run()
