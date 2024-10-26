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
    SsdReplicated = 0
    RandomPartitions = 1
    BalancedHnsw = 2
    BalancedLsh = 3

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="The name for the outputted dataset")
    parser.add_argument("--out", default="./out/", help="The path to the output directory")
    parser.add_argument("--architecture", default=["ssd", "random", "hnsw", "lsh"], nargs="*", help="The architecture according to which to partition the vector set.")
    parser.add_argument("--db", default="./data/deep1M/deep1M_query.fvecs", help="The path to the vector set that you want to partition")
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


def compute_random_partitions(data, num_partitions):
    # for i in range(len(data)):
    np.random.shuffle(data)
    return np.array_split(data, num_partitions)

def compute_balanced_partitions(data, num_partitions):
    print("Computing {} balanced partitions...", num_partitions)
    pass

def get_fname(architecture: Partitions, how_many: int, node_num: int, dataset_name: str):
    if architecture is Partitions.SsdReplicated:
        return f"{dataset_name}_{architecture.name}.fvecs"

    return f"{dataset_name}_{architecture.name}_{how_many}nodes_node{node_num}.fvecs"

def mkdir_p(directory_path):
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    print("Created {}.", directory_path)

def compute_partitions(data, how_many: int, schema: Partitions, out_dir: str, dataset_name: str):
    match schema:

        case Partitions.SsdReplicated:
            partition_name = get_fname(schema, how_many, 1, dataset_name)
            fvecs_write(Path(out_dir) / Path(partition_name), data)
            print("Wrote {}.", partition_name)
            # TODO: just create a file with the main fvecs
        case Partitions.RandomPartitions: 
            compute_random_partitions(data, how_many)
        case Partitions.BalancedHnsw:
            compute_balanced_partitions(data, how_many)
        case Partitions.BalancedLsh:
            compute_balanced_partitions(data, how_many)
        case _:
            print("Error: No such partition")

if __name__ == '__main__':
    args = process_args()

    mkdir_p(args.out)

    for arch in args.architecture:
        print("Calculating partitions for {}:".format(arch))
        partition_schema = {
            "ssd": Partitions.SsdReplicated,
            "random": Partitions.RandomPartitions,
            "lsh": Partitions.BalancedLsh,
            "hnsw": Partitions.BalancedHnsw,
        }[arch]

        base_db = fvecs_mmap(args.db)
        base_db = np.ascontiguousarray(base_db, dtype='float32')

        print("Base db shape: {}", base_db.shape)

        clustersizes = [1, 2, 5, 10]
        for how_many in clustersizes:
            compute_partitions(base_db, how_many, partition_schema, args.out, args.name)

        # out= str(Path(args.out) / "deep{}_groundtruth.ivecs".format(szsuf))
        # ivecs_write(gt_fname, gt)

