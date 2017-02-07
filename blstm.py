import tensorflow as tf


def get_cell(num_units, num_layers):
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
    if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
    return cell


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
                 dtype=tf.float32):

        self.inputs = tf.placeholder(tf.int32, [None, num_steps])
        self.labels = tf.placeholder(tf.int32, [None, num_steps])
        self.lengths = tf.placeholder(tf.int32, [None])
        self.weights = tf.placeholder(dtype, [None, num_steps])

        embedding = tf.get_variable(
            'embedding', [vocab_size, emb_size], dtype=dtype)

        inp_emb = tf.nn.embedding_lookup(embedding, self.inputs)

        outputs, _, _ = tf.nn.bidirectional_rnn(
            cell_fw=get_cell(num_units, num_layers),
            cell_bw=get_cell(num_units, num_layers),
            inputs=tf.unpack(tf.transpose(inp_emb, perm=[1, 0, 2])),
            dtype=dtype,
            sequence_length=self.lengths)

        output = tf.reshape(
            tf.transpose(tf.pack(outputs), perm=[1, 0, 2]), [-1, 2 * num_units])

        W = tf.get_variable(
            'W', [num_units * 2, num_labels], dtype=dtype,
            initializer=tf.truncated_normal_initializer(stddev=.01))
        b = tf.get_variable(
            'b', [num_labels], dtype=dtype,
            initializer=tf.constant_initializer(.1))

        logits = tf.reshape(
            tf.matmul(output, W) + b, [-1, num_steps, num_labels])

        self.loss = tf.nn.seq2seq.sequence_loss(
            tf.unstack(logits, axis=1),
            tf.unstack(self.labels, axis=1),
            tf.unstack(self.weights, axis=1))

        optimizer = tf.train.AdamOptimizer(learning_rate)
        params = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, params), max_clip_norm)

        self.probs = tf.nn.softmax(logits)
        self.train = optimizer.apply_gradients(zip(grads, params))
        self.saver = tf.train.Saver(tf.global_variables())
