#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
from collections import namedtuple
from tensorflow.python.client import timeline


class GraphOp(namedtuple('GraphOp', ['out_op', 'state_op'])):
    pass


class DatasetSynth:
    def __init__(self, config):
        self._batch_size = config.batchSize
        self._num_features = config.nFeatures
        self._dtype = tf.float32 if config.type == 32 else tf.float32
        self._max_time = config.nSteps
        self._n_batch = config.nBatches
        self._iter = self._create_data_iter()
        self._next = self._iter.get_next()

    @property
    def iter(self):
        return self._iter

    @property
    def next(self):
        return self._next

    def _create_data_iter(self):
        n_all = self._n_batch * self._batch_size
        dtype = np.float32 if self._dtype == tf.float32 else np.float64
        data = np.random.rand(n_all, self._max_time, self._num_features).astype(dtype)
        dataset = tf.data.Dataset.from_tensor_slices(data).batch(self._batch_size)
        return dataset.make_initializable_iterator()

class LSTMTest:
    def __init__(self, inputs, config, is_cudnn=False, is_training=False):
        self._config = config
        self._hidden_size = config.nHiddenUnits
        self._n_features = config.nFeatures
        self._num_layers = config.nLayers
        self._batch_size = config.batchSize
        self._inputs = inputs
        self._type = tf.float32 if config.type == 32 else tf.float64
        self._init_scale = 1.
        self._graph = self._get_graph_ops(is_cudnn, is_training)

    @property
    def config(self):
        return self._config

    @property
    def graph(self):
        return self._graph

    def _build_rnn_graph_cudnn(self, inputs, is_training):
        """Build the inference graph using CUDNN cell."""
        inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=self._num_layers,
        num_units=self._hidden_size)
        # zero is provided in this case
        outputs, (h, c) = self._cell(inputs, training=is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, self._hidden_size])
        return GraphOp(outputs, (tf.nn.rnn_cell.LSTMStateTuple(h=h, c=c)))

    def _get_lstm_cell(self):
        return tf.nn.rnn_cell.BasicLSTMCell(
        self._hidden_size, forget_bias=1.0, state_is_tuple=True)

    def _build_rnn_graph_tensorflow(self, inputs):
        def make_cell():
            return self._get_lstm_cell()
        cell = tf.nn.rnn_cell.MultiRNNCell(
        [make_cell() for _ in range(self._num_layers)], state_is_tuple=True)
        self._initial_state = cell.zero_state(self._batch_size, self._type)
        outputs, state = tf.nn.dynamic_rnn(cell, inputs,
            initial_state=self._initial_state)
        return GraphOp(outputs, state)

    def _get_graph_ops(self, is_cudnn=False, is_training=False):
        if is_cudnn:
            return self._build_rnn_graph_cudnn(self._inputs, is_training)
        else:
            return self._build_rnn_graph_tensorflow(self._inputs)

    def run_graph(self, sess, data):
        return sess.run(self._graph.out_op, feed_dict={self._inputs: data})


def run_profiler(sess, out_op, step, timeline_fname):
    profiler = tf.profiler.Profiler(sess.graph)
    run_opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(out_op, options=run_opts, run_metadata=run_metadata)
    profiler.add_step(step, run_metadata)
    opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
    profiler.profile_operations(options=opts)

    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    serialized_trace = fetched_timeline.generate_chrome_trace_format()
    with open(timeline_fname, 'w') as flog:
        flog.write(serialized_trace)

    # profiler.profile_graph(options=opts)
    # profiler.profile_name_scope(options=opts)
    # with open('test.log', 'w') as f:
    #   f.write(profiler.serialize_to_string())


if __name__ == '__main__':

    num_layers = 2
    num_units = 1024
    num_features = 1024
    dtype = 32
    max_time = 8
    n_epochs = 1
    n_batches = 1
    batch_size = 1
    log_dir = './logs/'
    gpu_timeline_f = log_dir + 'gpu_lstm_timeline.json'

    Config = namedtuple('Config', ['nHiddenUnits', 'nFeatures',
        'nLayers', 'batchSize', 'nSteps', 'nBatches', 'type'])
    config = Config(num_units, num_features, num_layers, batch_size, max_time, n_batches, dtype)
    dataset = DatasetSynth(config)

    with tf.Session() as sess:
        # inputs = tf.placeholder(tf.float32, shape=[_batch_size, max_time, num_features])
        lstm = LSTMTest(dataset.next, config, is_cudnn=True)

        sess.run(tf.global_variables_initializer())
        for _ in range(n_epochs):
            sess.run(dataset.iter.initializer)
            step = 0
        while True:
            try:
                run_profiler(sess, lstm.graph.out_op, step, gpu_timeline_f)
                step += 1
            except tf.errors.OutOfRangeError:
                break
