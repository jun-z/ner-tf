from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import os
import glob
import numpy as np
import tensorflow as tf

from blstm import BLSTM
from utils import load_data, accuracy
from sklearn.utils import shuffle

tf.app.flags.DEFINE_string('data_dir', './data', 'Data directory.')
tf.app.flags.DEFINE_string('train_dir', './model', 'Training directory.')
tf.app.flags.DEFINE_integer('num_epochs', 5, 'Number of epochs to train.')
tf.app.flags.DEFINE_integer('batch_size', 20, 'Batch size.')
tf.app.flags.DEFINE_integer('num_units', 50, 'Number of units in LSTM.')
tf.app.flags.DEFINE_integer('num_layers', 1, 'Number of LSTM layers.')
tf.app.flags.DEFINE_integer('num_steps', 25, 'Max number of time steps')
tf.app.flags.DEFINE_integer('num_labels', 8, 'Number of labels.')
tf.app.flags.DEFINE_integer('emb_size', 10, 'Size of embedding.')
tf.app.flags.DEFINE_integer('vocab_size', 55, 'Size of vocabulary.')
tf.app.flags.DEFINE_integer('es_patience', 0, 'Patience for early stopping.')
tf.app.flags.DEFINE_float('learning_rate', .03, 'Learning rate.')
tf.app.flags.DEFINE_float('max_clip_norm', 5.0, 'Clip norm for gradients.')
tf.app.flags.DEFINE_bool('save_best_only', True, 'Save only best models.')
tf.app.flags.DEFINE_bool('trainable_embs', True, 'Trainable embeddings.')
tf.app.flags.DEFINE_bool('pretrained_embs', False, 'Pretrained embeddings.')
tf.app.flags.DEFINE_bool('use_crf', False, 'Use CRF loss.')
tf.app.flags.DEFINE_bool('use_fp16', False, 'Use tf.float16.')

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
        get_embs(),
        FLAGS.trainable_embs,
        tf.float16 if FLAGS.use_fp16 else tf.float32)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Restoring model from %s.' % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        print('Created model with fresh parameters.')
        session.run(tf.global_variables_initializer())
        epoch = 0
    return epoch, model


def get_embs():
    if FLAGS.pretrained_embs:
        emb_files = glob.glob(os.path.join(FLAGS.data_dir, '*.vec'))

        if len(emb_files) == 0:
            raise Exception('Did not find file with .vec ext.')
        elif len(emb_files) > 1:
            raise Exception('Found more than one file with .vec ext.')

        print('Loading embeddings from %s.' % emb_files[0])

        embs = []
        with open(emb_files[0]) as f:
            for i, line in enumerate(f):
                if i == 0:
                    vocab_size, emb_size = [int(s) for s in line.split()]
                    scale = np.sqrt(3) / np.sqrt(vocab_size + 2)
                    for i in range(2):
                        rand = np.random.uniform(-scale, scale, emb_size)
                        embs.append(rand.astype('float32'))
                else:
                    embs.append(np.asarray(line.split()[1:], dtype='float32'))

        return np.vstack(embs)
    else:
        return None


def train():
    train, valid, test = load_data(FLAGS.data_dir)

    if not os.path.exists(FLAGS.train_dir):
        print('Directory %s does not exist, creating...' % FLAGS.train_dir)
        os.makedirs(FLAGS.train_dir)

    with tf.Session() as sess:
        epoch, model = create_model(sess)
        valid_losses = []
        es_count = 0

        for i in range(FLAGS.num_epochs):
            train_losses = []
            tokens, labels, lengths, weights = shuffle(
                train['tokens'], train['labels'],
                train['lengths'], train['weights'])

            for p in range(0, len(tokens), FLAGS.batch_size):
                loss, _ = sess.run(
                    [model.loss, model.train],
                    feed_dict={
                        model.tokens: tokens[p:p + FLAGS.batch_size],
                        model.labels: labels[p:p + FLAGS.batch_size],
                        model.lengths: lengths[p:p + FLAGS.batch_size],
                        model.weights: weights[p:p + FLAGS.batch_size]})
                train_losses.append(loss)

            if FLAGS.use_crf:
                loss, logits, trans_params = sess.run(
                    [model.loss, model.logits, model.trans_params],
                    feed_dict={
                        model.tokens: valid['tokens'],
                        model.labels: valid['labels'],
                        model.lengths: valid['lengths'],
                        model.weights: valid['weights']})

                preds = [
                    tf.contrib.crf.viterbi_decode(logits[l], trans_params)[0]
                    for l in range(logits.shape[0])]
            else:
                loss, probs = sess.run(
                    [model.loss, model.probs], feed_dict={
                        model.tokens: valid['tokens'],
                        model.labels: valid['labels'],
                        model.lengths: valid['lengths'],
                        model.weights: valid['weights']})

                preds = np.argmax(probs, axis=2)

            valid_losses.append(loss)
            l_acc, r_acc = accuracy(preds, valid['labels'], valid['lengths'])
            print('Epoch %i finished' % (epoch + i + 1))
            print('* final training loss %0.2f' % train_losses[-1])
            print('* validation loss %0.2f' % loss)
            print('* label level accuracy %0.2f' % l_acc)
            print('* record level accuracy %0.2f' % r_acc)

            if FLAGS.save_best_only and len(valid_losses) >= 2:
                if valid_losses[-1] < min(valid_losses[:-1]):
                    model.saver.save(
                        sess,
                        os.path.join(FLAGS.train_dir, 'ner.ckpt'),
                        global_step=(epoch + i + 1))
            else:
                model.saver.save(
                    sess,
                    os.path.join(FLAGS.train_dir, 'ner.ckpt'),
                    global_step=(epoch + i + 1))

            if FLAGS.es_patience > 0 and len(valid_losses) >= 2:
                if valid_losses[-1] >= valid_losses[-2]:
                    es_count += 1
                else:
                    es_count = 0
                if es_count == FLAGS.es_patience:
                    print('Validation loss stopped decreasing.')
                    print('Stopping training...')
                    break

        if FLAGS.use_crf:
            logits, trans_params = sess.run(
                [model.logits, model.trans_params], feed_dict={
                    model.tokens: test['tokens'],
                    model.labels: test['labels'],
                    model.lengths: test['lengths'],
                    model.weights: test['weights']})

            preds = [
                tf.contrib.crf.viterbi_decode(logits[l], trans_params)[0]
                for l in range(logits.shape[0])]
        else:
            loss, probs = sess.run(
                [model.loss, model.probs], feed_dict={
                    model.tokens: test['tokens'],
                    model.labels: test['labels'],
                    model.lengths: test['lengths'],
                    model.weights: test['weights']})

            preds = np.argmax(probs, axis=2)

        l_acc, r_acc = accuracy(preds, test['labels'], test['lengths'])
        print('Training finished')
        print('* testing loss %0.2f' % loss)
        print('* label level accuracy %0.2f' % l_acc)
        print('* record level accuracy %0.2f' % r_acc)


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
