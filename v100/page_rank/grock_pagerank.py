from ctypes import *
import time
import logging


def run_grock_pr(_indptr_fname, _indices_fname, grock_dir, _log_dir):
    logging.basicConfig(filename=_log_dir)
    logging.getLogger().setLevel(logging.INFO)

    grock = cdll.LoadLibrary(grock_dir)
    row_list = [int(x.strip()) for x in open(_indptr_fname)]
    col_list = [int(x.strip()) for x in open(_indices_fname)]
    print("number of ptrs = {}".format(len(row_list)))
    print("number of indices = {}".format(len(col_list)))

    row = pointer((c_int * len(row_list))(*row_list))
    col = pointer((c_int * len(col_list))(*col_list))
    nodes = len(row_list) - 1
    edges = len(col_list)
    node = pointer((c_int * nodes)())
    rank = pointer((c_float * nodes)())
    normalize = 1

    logging.warning("Running gunrock pagerank with default damping factor = 0.85...")
    start_time = time.time()
    grock_elapsed = grock.pagerank(nodes, edges, row, col, normalize, node, rank)
    host_elapsed = time.time() - start_time
    logging.info("Elapsed time measured by gunrock: {}s.".format(str(grock_elapsed)))
    logging.info("Elapsed time measured by host: {}s.".format(host_elapsed))

    # TODO: Need to figure out a way to store LP_c_float_Array to file system.


if __name__ == '__main__':
    src_name = './data/delaunay_n20'
    indptr_fname = src_name + '_ofst.csv'
    indices_fname = src_name + '_edges.csv'
    grock_lib_dir = './grock/build/lib/libgunrock.so'
    log_dir = './logs/'
    gpu_timeline_f = log_dir + 'gpu_grock_pr.log'
    run_grock_pr(indptr_fname, indices_fname, grock_lib_dir, gpu_timeline_f)
