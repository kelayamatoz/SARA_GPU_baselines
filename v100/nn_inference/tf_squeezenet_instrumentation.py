#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf
import numpy as np
import logging
from tensorflow.python.client import timeline
from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary

try:
    from tensorflow.contrib.tensorrt.ops.gen_trt_engine_op import *
except ImportError:
    pass


def save_graph_with_suffix(sess, _log_dir, suffix):
    pb_visual_writer = summary.FileWriter(_log_dir, filename_suffix=suffix)
    pb_visual_writer.add_graph(sess.graph)


def test_inference(
        _batch_size, _model_dir, timeline_fname, logging_fname,
        _session_conf=None, _n_test_iters=64, _n_burn_iters=4
):
    logging.basicConfig(filename=logging_fname)
    logging.getLogger().setLevel(logging.INFO)

    is_throughput_test = _batch_size != 1
    with session.Session(graph=ops.Graph(), config=_session_conf) as sess:
        with gfile.GFile(_model_dir, 'rb') as f:
            graph_def = graph_pb2.GraphDef()
            graph_def.ParseFromString(f.read())
            importer.import_graph_def(graph_def)
            sess.graph.as_default()

            options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()
            g = sess.graph
            output_tensor = g.get_tensor_by_name('import/ArgMax:0')
            np_val = np.random.rand(_batch_size, H, W, C)

            if is_throughput_test:
                logging.info("Starting throughput test.")
                logging.info(
                    "batch_size = {}, n_test_iters = {}, n_burn_iters = {}.".format(
                        _batch_size, _n_test_iters, _n_burn_iters
                    )
                )
                input_tensor = g.get_tensor_by_name('import/Placeholder:0')

                tf_bench = tf.test.Benchmark()
                throughput_metrics = _batch_size
                # MBs refers to the number of bytes moved by this op.
                # In our case, we use the number of images moved by this op.
                report = tf_bench.run_op_benchmark(
                    sess,
                    output_tensor,
                    feed_dict={input_tensor: np_val},
                    burn_iters=_n_burn_iters,
                    min_iters=_n_test_iters,
                    store_trace=False,
                    mbs=_batch_size
                )
                # Using 15623MiB / 16130MiB
                s_wall_time = report['wall_time']

                # Throughput is in n_images / s
                throughput = _batch_size / float(s_wall_time)
                logging.info("Calculated throughput = {} images / s".format(throughput))

            else:
                logging.info("Running inference test")
                try:
                    input_tensor = g.get_tensor_by_name("import/Placeholder:0")
                    _ = sess.run(
                        output_tensor, feed_dict={input_tensor: np_val}, options=options, run_metadata=run_metadata
                    )
                except KeyError:
                    _ = sess.run(
                        output_tensor, feed_dict=dict(), options=options, run_metadata=run_metadata
                    )

                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                serialized_trace = fetched_timeline.generate_chrome_trace_format()
                with open(timeline_fname, 'w') as flog:
                    flog.write(serialized_trace)


if __name__ == '__main__':
    # test_inference(1, gpu_profiling_model_name, './logs/cpu_timeline_no_placeholder.json', _session_conf)

    model_dir = './nn_inference/squeezenet_tflite_pretrained/squeezenet.pb'
    log_dir = './logs/'
    H = 224
    W = 224
    C = 3
    input_name = 'Placeholder'

    batch_size = 1
    n_experiments = 10
    use_gpu = False
    filename_suffix = '_raw_graph_new'
    gpu_profiling_model_name = 'gpu_profiling_squeezenet.pb'
    logging_f = log_dir + 'gpu_squeezenet_throughput_info.log'
    cpu_timeline_f = 'cpu_time_line.json'
    gpu_timeline_f = log_dir + 'gpu_sqznet_timeline.json'

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # test_inference(1, model_dir, gpu_timeline_f, session_conf)

    throughput_test_iters = 256
    batch_size = 512
    test_inference(
        batch_size, model_dir, gpu_timeline_f, logging_f,
        _session_conf=session_conf, _n_burn_iters=4, _n_test_iters=throughput_test_iters
    )

