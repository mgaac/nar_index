import mlx.core as mx

import numpy as np
import pickle
import os

def pickle_read(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def generate_connection_matrix(graph):
    edges = [(src, trg) for src, trgs in graph.items() for trg in trgs]
    src_idx, trg_idx = zip(*edges)

    return mx.array([src_idx, trg_idx])

def load_data(data_dir: str = 'data/CORA'):
    allx = pickle_read(os.path.join(data_dir, 'allx'))
    ally = pickle_read(os.path.join(data_dir, 'ally'))
    tx = pickle_read(os.path.join(data_dir, 'tx'))
    ty = pickle_read(os.path.join(data_dir, 'ty'))

    graph = pickle_read(os.path.join(data_dir, 'graph'))

    test_idx = np.loadtxt(os.path.join(data_dir, 'test_index'), dtype=int)

    allx = mx.array(allx.todense())
    ally = mx.array(ally)
    tx = mx.array(tx.todense())
    ty = mx.array(ty)

    test_idx = mx.array(test_idx)

    node_embeddings = mx.concatenate([allx, tx], axis=0)
    labels = mx.concat([ally, ty], axis=0)
    connection_matrix = generate_connection_matrix(graph)


    train_mask = mx.ones([labels.shape[0]]).at[test_idx].add(-1)
    test_mask = mx.zeros([labels.shape[0]]).at[test_idx].add(1)

    return node_embeddings, connection_matrix, labels, train_mask, test_mask
