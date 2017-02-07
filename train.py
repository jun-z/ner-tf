from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import os
import numpy as np
import tensorflow as tf

from blstm import BLSTM
from utils import load_data, load_vocab, accuracy

tf.app.flags.DEFINE_string('data_dir', './data', 'Data directory.')
tf.app.flags.DEFINE_string('train_dir', './model', 'Training directory.')
tf.app.flags.DEFINE_integer('num_epochs', 5, 'Number of epochs.')
tf.app.flags.DEFINE_integer('batch_size', 20, 'Batch size.')
tf.app.flags.DEFINE_integer('num_units', 50, 'Number of units in LSTM.')
tf.app.flags.DEFINE_integer('num_layers', 1, 'Number of LSTM layers.')
tf.app.flags.DEFINE_integer('num_steps', 25, 'Max number of time steps')
tf.app.flags.DEFINE_integer('num_labels', 8, 'Number of labels.')
tf.app.flags.DEFINE_integer('emb_size', 10, 'Size of embedding.')
tf.app.flags.DEFINE_integer('vocab_size', 45, 'Size of vocabulary.')
tf.app.flags.DEFINE_float('learning_rate', .03, 'Learning rate.')
tf.app.flags.DEFINE_bool('use_fp16', False, 'Use tf.float16.')
tf.app.flags.DEFINE_bool('do_label', False, 'Train model or label sequence.')

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
        tf.float16 if FLAGS.use_fp16 else tf.float32)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Restoring model from %s.' % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print('Created model with fresh parameters.')
        session.run(tf.global_variables_initializer())
    return model


def train():
    train, valid, test = load_data(FLAGS.data_dir)

    with tf.Session() as sess:
        model = create_model(sess)
        batch_size = FLAGS.batch_size
        for epoch in range(FLAGS.num_epochs):
            for p in range(0, len(train['inputs']), batch_size):
                sess.run(
                    model.train,
                    feed_dict={
                        model.inputs: train['inputs'][p:p + batch_size],
                        model.labels: train['labels'][p:p + batch_size],
                        model.lengths: train['lengths'][p:p + batch_size],
                        model.weights: train['weights'][p:p + batch_size]})
            model.saver.save(
                sess,
                os.path.join(FLAGS.train_dir, 'ner.ckpt'),
                global_step=epoch)
            loss, probs = sess.run(
                [model.loss, model.probs], feed_dict={
                    model.inputs: valid['inputs'],
                    model.labels: valid['labels'],
                    model.lengths: valid['lengths'],
                    model.weights: valid['weights']})
            preds = np.argmax(probs, axis=2)
            l_acc, r_acc = accuracy(preds, valid['labels'], valid['lengths'])
            print('Epoch %i finished' % epoch)
            print('* validation loss %0.2f' % loss)
            print('* label level accuracy %0.2f' % l_acc)
            print('* record level accuracy %0.2f' % r_acc)
        loss, probs = sess.run(
            [model.loss, model.probs], feed_dict={
                model.inputs: test['inputs'],
                model.labels: test['labels'],
                model.lengths: test['lengths'],
                model.weights: test['weights']})
        preds = np.argmax(probs, axis=2)
        l_acc, r_acc = accuracy(preds, test['labels'], test['lengths'])
        print('Training finished')
        print('* testing loss %0.2f' % loss)
        print('* label level accuracy %0.2f' % l_acc)
        print('* record level accuracy %0.2f' % r_acc)


def label():
    vocab = load_vocab(FLAGS.data_dir, '.vocab')
    labels = load_vocab(FLAGS.data_dir, '.labels')
    with tf.Session() as sess:
        model = create_model(sess)
        while True:
            tokens = raw_input('Please enter query: ').split()
            inputs = []
            for t in tokens:
                if t in vocab:
                    inputs.append(vocab.index(t))
                else:
                    inputs.append(1)
            length = len(inputs)
            inputs += [0] * (FLAGS.num_steps - length)
            probs = sess.run(model.probs,
                             feed_dict={
                                 model.inputs: [inputs],
                                 model.lengths: [length]
                             })
            preds = np.argmax(probs, axis=2)[0]
            for i in range(length):
                print(tokens[i], '->', labels[preds[i]])


def main(_):
    if FLAGS.do_label:
        label()
    else:
        train()


if __name__ == '__main__':
    tf.app.run()
