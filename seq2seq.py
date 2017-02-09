import tensorflow as tf


class Seq2Seq(object):
    def __init__(self,
                 num_units,
                 num_layers,
                 source_steps,
                 target_steps,
                 source_vocab_size,
                 target_vocab_size,
                 emb_size,
                 learning_rate,
                 max_clip_norm,
                 dtype=tf.float32):

        self.encoder_inputs = tf.placeholder(tf.int32, [None, source_steps])
        self.decoder_inputs = tf.placeholder(tf.int32, [None, target_steps])

        self.encoder_length = tf.placeholder(tf.int32, [None])
        self.decoder_length = tf.placeholder(tf.int32, [None])

        self.target_weights = tf.placeholder(dtype, [None, target_steps - 1])

        embedding = tf.get_variable(
            'embedding', [source_vocab_size, emb_size], dtype=dtype)

        inp_emb = tf.nn.embedding_lookup(embedding, self.encoder_inputs)

        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

        encoder_outputs, encoder_state = tf.nn.rnn(
            cell, tf.unstack(inp_emb, axis=1), dtype=dtype,
            sequence_length=self.encoder_length, scope='encoder')

        W = tf.get_variable(
            'W', [num_units, target_vocab_size], dtype=dtype,
            initializer=tf.truncated_normal_initializer(stddev=.01))
        b = tf.get_variable(
            'b', [target_vocab_size], dtype=dtype,
            initializer=tf.constant_initializer(.1))

        one_hot = tf.one_hot(self.decoder_inputs, target_vocab_size)

        decoder_outputs, decoder_state = tf.nn.rnn(
            cell, tf.unstack(one_hot, axis=1), encoder_state, dtype=dtype,
            sequence_length=self.decoder_length, scope='decoder')

        outputs = tf.reshape(
            tf.pack(decoder_outputs, axis=1), [-1, num_units])

        logits = tf.reshape(
            tf.matmul(outputs, W) + b, [-1, target_steps, target_vocab_size])

        self.loss = tf.nn.seq2seq.sequence_loss(
            tf.unstack(logits, axis=1)[:-1],
            tf.unstack(self.decoder_inputs, axis=1)[1:],
            tf.unstack(self.target_weights, axis=1))

        with tf.variable_scope('decoder', reuse=True):
            out = tf.arg_max(tf.unstack(logits, axis=1)[0], 1)
            pre = tf.one_hot(out, target_vocab_size)
            state = decoder_state
            preds = [out]
            for i in range(target_steps - 2):
                out, state = cell(pre, state)
                preds.append(tf.arg_max(tf.matmul(out, W) + b, 1))
                pre = tf.one_hot(preds[-1], target_vocab_size)
            self.preds = tf.pack(preds, axis=1)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        params = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, params), max_clip_norm)

        self.train = optimizer.apply_gradients(zip(grads, params))
        self.saver = tf.train.Saver(tf.global_variables())
