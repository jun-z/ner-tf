import tensorflow as tf


class BLSTM(object):
    def __init__(self,
                 num_units,
                 num_layers,
                 num_steps,
                 num_labels,
                 emb_size,
                 vocab_size,
                 learning_rate,
                 max_clip_norm,
                 use_crf,
                 dtype=tf.float32):

        self.inputs = tf.placeholder(tf.int32, [None, num_steps])
        self.labels = tf.placeholder(tf.int32, [None, num_steps])
        self.lengths = tf.placeholder(tf.int32, [None])
        self.weights = tf.placeholder(dtype, [None, num_steps])

        with tf.device('/cpu:0'):
            embedding = tf.get_variable(
                'embedding', [vocab_size, emb_size], dtype=dtype)

            inp_emb = tf.nn.embedding_lookup(embedding, self.inputs)

        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

        outputs, _, _ = tf.nn.bidirectional_rnn(
            cell_fw=cell,
            cell_bw=cell,
            inputs=tf.unstack(inp_emb, axis=1),
            dtype=dtype,
            sequence_length=self.lengths)

        output = tf.reshape(tf.stack(outputs, axis=1), [-1, 2 * num_units])

        with tf.variable_scope('Projection'):
            W = tf.get_variable(
                'W', [num_units * 2, num_labels], dtype=dtype,
                initializer=tf.truncated_normal_initializer(stddev=.01))
            b = tf.get_variable(
                'b', [num_labels], dtype=dtype,
                initializer=tf.constant_initializer(.1))

        self.logits = tf.reshape(
            tf.matmul(output, W) + b, [-1, num_steps, num_labels])

        if use_crf:
            ll, self.trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, sequence_lengths=self.lengths)
            self.loss = tf.reduce_mean(-ll)
        else:
            self.loss = tf.nn.seq2seq.sequence_loss(
                tf.unstack(self.logits, axis=1),
                tf.unstack(self.labels, axis=1),
                tf.unstack(self.weights, axis=1))

            self.probs = tf.nn.softmax(self.logits)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        params = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, params), max_clip_norm)

        self.train = optimizer.apply_gradients(zip(grads, params))
        self.saver = tf.train.Saver(tf.global_variables())
