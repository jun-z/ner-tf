from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import cPickle as pickle
import tensorflow as tf

from blstm import BLSTM
from utils import load_vocab

tf.app.flags.DEFINE_string('input', '', 'Input file.')
tf.app.flags.DEFINE_string('output', '', 'Output file.')
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
tf.app.flags.DEFINE_bool('use_fp16', False, 'Use tf.float16.')
tf.app.flags.DEFINE_bool('flat_file', False, 'Flat file or pickled.')
tf.app.flags.DEFINE_bool('diff_only', False, 'Output only differences.')

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
        tf.float16 if FLAGS.use_fp16 else tf.float32)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Restoring model from %s.' % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        return model
    else:
        raise Exception('Could not find model in %s'
                        % ckpt.model_checkpoint_path)


def write_record(f, inputs, true_labels, pred_labels):
    for i, t, p in zip(inputs, true_labels, pred_labels):
        f.write(' '.join([i, t, p]) + '\n')
    f.write('\n')


def eval():
    vocab_l = load_vocab(FLAGS.data_dir, '.vocab')
    labels_l = load_vocab(FLAGS.data_dir, '.labels')
    if FLAGS.flat_file:
        pass
    else:
        with open(FLAGS.input, 'rb') as f:
            data = pickle.load(f)

        with tf.Session() as sess:
            model = create_model(sess)
            probs = sess.run(
                model.probs, feed_dict={
                    model.inputs: data['inputs'],
                    model.lengths: data['lengths']
                })
            preds = np.argmax(probs, axis=2)
            with open(FLAGS.output, 'w') as f:
                for i in range(preds.shape[0]):
                    correct = True
                    inputs = []
                    true_labels = []
                    pred_labels = []
                    for j in range(data['lengths'][i]):
                        pred_label = preds[i][j]
                        true_label = data['labels'][i][j]
                        inputs.append(vocab_l[data['inputs'][i][j]])
                        true_labels.append(labels_l[true_label])
                        pred_labels.append(labels_l[pred_label])
                        if preds[i][j] != data['labels'][i][j]:
                            correct = False
                    if correct:
                        if not FLAGS.diff_only:
                            write_record(f, inputs, true_labels, pred_labels)
                    else:
                        write_record(f, inputs, true_labels, pred_labels)


def main(_):
    eval()


if __name__ == '__main__':
    tf.app.run()
