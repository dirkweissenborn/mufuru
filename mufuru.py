import tensorflow as tf
from tensorflow.python.ops.rnn.rnn_cell import *

_operations = {"max": lambda s, v: tf.maximum(s, v),
               "keep": lambda s, v: s,
               "replace": lambda s, v: v,
               "mul": lambda s, v: tf.mul(s, v),
               "min": lambda s, v: tf.minimum(s, v),
               "diff": lambda s, v: 0.5 * tf.abs(s - v),
               "forget": lambda s, v: tf.zeros_like(s),
               "sqr_diff": lambda s, v: 0.25 * (s - v)**2}


class MuFuRUCell(RNNCell):

    def __init__(self, num_units, op_controller_size=None,
                 ops=(_operations["keep"], _operations["replace"], _operations["mul"]),
                 op_biases=None):
        """
        :param num_units: number of hidden units
        :param op_controller_size: if > 0 then use of recurrent controller for computing operation weights
        :param ops: list of operations as python function objects with input parameters s
                    (representing the old memory state) and v (representing the newly computed feature vector)
        :param op_biases: optional, can be used to set initial bias on specific operations
        """
        self._num_units = num_units
        self._op_controller_size = 0 if op_controller_size is None else op_controller_size
        self._op_biases = list(op_biases)
        self._ops = ops if ops is not None else list(map(lambda _: 0.0, ops))
        self._num_ops = len(ops)

    @staticmethod
    def from_op_names(operations, num_units, biases=None, op_controller_size=None):
        """
        factory method to create MuFuRU from operation names
        :param operations: list of names of operations from following:
                           "max", "keep", "replace", "mul", "min", "diff", "forget", "sqr_diff"
        :param num_units: number of hidden units
        :param biases:  optional, can be used to set initial bias on specific operations
        :param op_controller_size:
        :return: MuFuRUCell
        """
        if biases is None:
            biases = map(lambda _: 0.0, operations)
        assert len(list(biases)) == len(operations), "Operations and operation biases have to have same length."
        ops = list(map(lambda op: _operations[op], operations))
        return MuFuRUCell(num_units, op_controller_size, ops, biases)

    def _op_weights(self, inputs):
        # compute unnormalized operation weights
        t = tf.contrib.layers.fully_connected(inputs, self._num_units * self._num_ops, activation_fn=None)
        # compute softmax, using tf.nn.softmax was much slower than the following code
        weights = tf.split(1, self._num_ops, t)
        for i, w in enumerate(weights):
            if self._op_biases and self._op_biases[i] != 0.0:
                weights[i] = tf.exp((w + self._op_biases[i]))
            else:
                weights[i] = tf.exp(w)
        acc = tf.add_n(weights)
        weights = [tf.div(weights[i], acc, name="op_weight_%d" % i) for i in range(len(weights))]
        return weights

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units + max(self._op_controller_size, 0)

    def __call__(self, inputs, state, scope=None):
        """The Multi-Function Recurrent Unit (MuFuRUCell)"""
        with vs.variable_scope(scope or type(self).__name__):
            s, op_ctr = None, None
            if self._op_controller_size > 0:
                op_ctr = tf.slice(state, [0, 0], [-1, self._op_controller_size])
                s = tf.slice(state, [0, self._op_controller_size], [-1, self._num_units])
            else:
                s = state
            with vs.variable_scope("Gate"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not udpate.
                r = tf.contrib.layers.fully_connected(tf.concat(1, [inputs, s]), self._num_units,
                                                      activation_fn=tf.nn.sigmoid,
                                                      biases_initializer=tf.constant_initializer(1.0))
            with vs.variable_scope("Feature"):
                f = tf.contrib.layers.fully_connected(tf.concat(1, [inputs, r * s]), activation_fn=tf.nn.sigmoid)
            new_op_ctr = None
            if self._op_controller_size > 0:
                with vs.variable_scope("Op_controller"):
                    # ReLU activation
                    new_op_ctr = tf.contrib.layers.fully_connected(tf.concat(1, [inputs, s, op_ctr]),
                                                                   self._op_controller_size)
            else:
                new_op_ctr = tf.concat(1, [inputs, s])
            with vs.variable_scope("Op"):
                # compute operation weights
                op_weights = self._op_weights(new_op_ctr)
                # compute weighted features
                new_cs = [o(s, f) * w for (o, w) in zip(self._ops, op_weights)]
                new_c = tf.add_n(new_cs)
        if self._op_controller_size > 0:
            # include also controller within recurrent state
            return new_c, tf.concat(1, [new_op_ctr, new_c])
        else:
            return new_c, new_c
