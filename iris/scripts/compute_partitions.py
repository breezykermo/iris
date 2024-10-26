# Some IO codes are from:
# - https://github.com/facebookresearch/faiss/blob/master/benchs/link_and_code/datasets.py
# - https://github.com/facebookresearch/faiss/blob/master/contrib/vecs_io.py

import faiss
import numpy as np
import argparse
from pathlib import Path
from enum import Enum, auto

# This represents all the architectures and balancing strategies we will test against
class Partitions(Enum):
    SSD1N = 1
    SSD2N = 2
    SSD5N = 3
    SSD10N = 4
    Random1N = 5
    Random2N = 6
    Random5N = 7
    Random10N = 8
    BalancedHNSW1N = 9 # check when you get benefits -> maybe 20 nodes
    BalancedHNSW2N = 10
    BalancedHNSW5N = 11
    BalancedHNSW10N = 12
    BalancedLSH1N = 13
    BalancedLSH2N = 14
    BalancedLSH5N = 15
    BalancedLSH10N = 16

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./", help="The path to the output directory")
    parser.add_argument("--szsufs", default=["1M", "10M", "100M"], nargs="*", help="The target sizes")
    parser.add_argument("--base_filename", default="./data/deep1M/deep1M_learn.fvecs", help="The path to the base veectors")
    parser.add_argument("--query_filename", default="./data/deep1M/deep1M_query.fvecs", help="The path to the query vectors")
    return parser.parse_args()

# Note: This is in memory. Use ivecs_mmap for the same functionality but with 
# memmap to be able to work with larger 
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

# This is what should be used for datasets that cannot be loaded into memory
# i.e., greater than 64GBs
def ivecs_mmap(fname):
    a = np.memmap(fname, dtype='int32', mode='r')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]


def fvecs_mmap(fname):
    return ivecs_mmap(fname).view('float32')


def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)

def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))


def do_compute_gt(xb, xq, topk=100):
    nb, d = xb.shape
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    _, ids = index.search(x=xq, k=topk)
    return ids.astype('int32')

def partition_to_nodes(data, num_nodes=4):
    """
    Partitions the data into 'num_nodes' equal parts and assigns to different nodes.
    """
    nb = data.shape[0]
    block_size = nb // num_nodes  # Determine the size of each partition
    
    partitions = []
    
    for node_id in range(num_nodes):
        start_idx = node_id * block_size
        # If last node, include all remaining vectors
        end_idx = nb if node_id == num_nodes - 1 else (node_id + 1) * block_size
        partitions.append(data[start_idx:end_idx])  # Partitioned data block
    
    return partitions 


def compute_SSD(data, num_partitions):
    '''
    We use this function to run experiments on
    - Single Node baseline where num_partitions = 1
    - Simple replication where, for n nodes, the data is simply copied
    into each of the n nodes.
    '''
    # Do we do our own filesystem to store this in?
    # Do we care if for smaller datasets this is effectively all in cache
    nodes = []
    for i in range(num_partitions):
        # {architecture}_{clustersize}nodes_node{nodenumber}.fvecs
        nodes.append(np.copy(data))
    pass

def compute_random_partitions(data, num_partitions):
    for i in range(len(data)):

    np.random.shuffle(data)
    
    return np.array_split(data, num_partitions)

def compute_balanced_partitions(data, num_partitions):

    pass

def compute_partitions(data, kind, how_many):
    match kind:
        case Partitions.SSD:
            compute_SSD(data, how_many)
        case Partitions.Random: # 2 nodes, 5 nodes
            compute_random_partitions(data, how_many)
        case Partitions.Balanced:
            compute_balanced_partitions(data, how_many)
        case _:
            print("Error: No such partition")
    pass


if __name__ == '__main__':
    args = process_args()

    for szsuf in args.szsufs:
        print("Deep{}:".format(szsuf))
        dbsize = {"1M": 1000000, "10M": 10000000, "100M": 100000000}[szsuf]

        # xb = mmap_fvecs('deep1b/base.fvecs')
        # xq = mmap_fvecs('deep1b/deep1B_queries.fvecs')
        # xt = mmap_fvecs('deep1b/learn.fvecs')
        # # deep1B's train is is outrageously big
        # xt = xt[:10 * 1000 * 1000]
        # 

        xb = fvecs_mmap(args.base_filename)
        xq = fvecs_read(args.query_filename)

        xb = np.ascontiguousarray(xb[:dbsize], dtype='float32')

        gt_fname = str(Path(args.out) / "deep{}_groundtruth.ivecs".format(szsuf))
        # gt = ivecs_read('deep1b/deep1B_groundtruth.ivecs') Added from FAISS
        gt = do_compute_gt(xb, xq, topk=100)

        ivecs_write(gt_fname, gt)

