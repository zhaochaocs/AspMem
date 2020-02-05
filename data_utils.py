import copy
from operator import itemgetter

import numpy as np
import torch


def batch_generator(dataset, batch_size, shuffle=True, mask=False):
    """
    Generates a batch iterator for a dataset.
    """
    data = dataset['data']
    data_original = dataset['original']
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    if shuffle:
        perm = np.random.permutation(data_size)
        data = itemgetter(*perm)(data)
        data_original = itemgetter(*perm)(data_original)

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)

        batched_data = copy.deepcopy(list(data[start_index:end_index]))
        batched_original = copy.deepcopy(list(data_original[start_index:end_index]))
        max_len_batch = len(max(batched_data, key=len))
        for j in range(len(batched_data)):
            batched_data[j].extend([0] * (max_len_batch - len(batched_data[j])))
        yield torch.from_numpy(np.array(batched_data)).long(), batched_original


def batch_test_generator(dataset, batch_size):
    """
    Generates a batch iterator for a dataset.
    """
    data = dataset['data']
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)

        batched_data = copy.deepcopy(list(data[start_index:end_index]))
        max_len_batch = len(max(batched_data, key=len))
        for j in range(len(batched_data)):
            batched_data[j].extend([0] * (max_len_batch - len(batched_data[j])))
        yield torch.from_numpy(np.array(batched_data)).long()

def load_aspects(aspect_seeds, word2id, w_emb_array):
    aspects_ids = []
    seed_weights = []

    with open(aspect_seeds, 'r') as fseed:
        for line in fseed:
            seeds = []
            weights = []
            for tok in line.split():
                word, weight = tok.split(':')
                if word in word2id:
                    seeds.append(word2id[word])
                    weights.append(float(weight))
                else:
                    seeds.append(0)
                    weights.append(0.0)
            aspects_ids.append(seeds)
            seed_weights.append(weights)

    seed_w = np.array(seed_weights)
    seed_w = seed_w / np.linalg.norm(seed_w, ord=1, axis=1, keepdims=True)  # 9 * 30
    seed_w = np.expand_dims(seed_w, axis=2)

    clouds = []
    for seeds in aspects_ids:
        clouds.append(w_emb_array[seeds])
    a_emb = np.array(clouds)  # 9,30,200
    a_emb = (a_emb * seed_w).sum(axis=1).astype(np.float32)  # 9, 200

    return a_emb