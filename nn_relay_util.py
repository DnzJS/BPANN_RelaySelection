import copy
import itertools
import multiprocessing
import random
import numpy as np
import time
import threading
import math
from multiprocessing.pool import ThreadPool


class ExpConfig:
    M = 8
    K = 2
    mu1 = 1
    mu2 = 2
    L = 2
    Pt = 1
    N0 = 1
    gamma_th = 1


class Solution:
    def __init__(self, raw_table, combi, config=None):
        if config is None:
            config = ExpConfig()
        self.combi = combi
        self.raw_table = raw_table
        # first max
        a = raw_table[combi[0]]
        b = raw_table[combi[1]]
        c = copy.deepcopy(a)
        for i in range(config.K):
            if c[i] < b[i]:
                c[i] = b[i]
        self.first_max = c
        # print(first_max)
        # second min
        self.second_min = np.min(self.first_max)


def get_outage_prob(data, result, gamma_th=None):
    config = ExpConfig()
    outage_count = 0
    for i in range(len(data)):
        tmp = []
        r = result[i]
        G = data[i]
        for m in range(config.M):
            if r[m] == 1:
                tmp.append(G[m])
        G_selected = np.max(np.array(tmp), axis=0)
        flag = config.Pt * np.min(G_selected) / config.N0

        if gamma_th is None:
            gamma_th = config.gamma_th

        if flag < gamma_th:
            outage_count += 1

    return outage_count / len(data)


def generate_G_idx():
    config = ExpConfig()
    G_idx = np.array(range(1,config.M * config.K + 1))
    random.shuffle(G_idx)
    G_min = G_idx.min()
    G_max = G_idx.max()
    # G_normalized = (G_idx - G_min) / (G_max - G_min)
    G_idx = np.reshape(G_idx, (8, 2))
    G_normalized = G_idx / G_max
    return G_idx, G_normalized


def get_max_values(G):
    config = ExpConfig()
    tmp_combi = itertools.combinations(range(config.M), 2)
    solutions = []
    for t in tmp_combi:
        solutions.append(Solution(G, t))
    max_solution = max(solutions, key=lambda x: x.second_min)
    max_values = max_solution.first_max
    return max_values


def get_data_max_values(data_size, config=None):
    if config is None:
        config = ExpConfig()
    data = []
    label = []
    t0 = time.time()
    for i in range(round(data_size)):
        idx, idx_norm = generate_G_idx()
        G_idx_full_reshape = np.reshape(idx_norm, [config.M*config.K])
        data.append(G_idx_full_reshape.tolist())
        tmp_label = get_max_values(idx_norm)
        # tmp_label = np.sort(tmp_label)
        label.append(tmp_label.tolist())
    t1 = time.time()
    gen_time = t1 - t0
    print("get_data: ", data_size, gen_time)
    return data, label


def get_relay_by_selection(G, s):
    config = ExpConfig()
    S_d_relay = [0] * config.M
    for i in range(config.K):
        dis = 9999
        selected_id = -1
        for j in range(config.M):
            if abs(s[i] - G[j][i]) < dis:
                dis = abs(s[i] - G[j][i])
                selected_id = j
        S_d_relay[selected_id] += 1

    for i in range(len(S_d_relay)):
        if S_d_relay[i] > 1:
            S_d_relay[i] = 1

    return S_d_relay


def multi_thread_gen(data_size=100, pool_size=-1):
    t0 = time.time()

    if pool_size == -1:
        pool_size = multiprocessing.cpu_count()
        pool_size = 1
    pool = multiprocessing.Pool(pool_size)
    print("pool_size: ", pool_size)

    data_size_per_thread = math.ceil(data_size / pool_size)
    t_results = pool.map(single_thread_gen, [data_size_per_thread] * pool_size)

    data = []
    label = []
    config = ExpConfig()
    for r in t_results:
        for x in r:
            idx_norm = x[0:8]
            G_idx_full_reshape = np.reshape(idx_norm, [config.M * config.K])
            data.append(G_idx_full_reshape.tolist())

            label.append(x[8])

    while len(data) > data_size:
        data.pop()
        label.pop()

    t1 = time.time()
    gen_time = t1 - t0
    print("get_data: ", data_size, gen_time)

    return data, label

def multi_thread_gen_x(data_size=100, pool_size=-1):
    t0 = time.time()

    if pool_size == -1:
        pool_size = multiprocessing.cpu_count() * 2
    pool = ThreadPool(pool_size)
    print("pool_size: ",pool_size)
    t_results = []
    data_size_per_thread = math.ceil(data_size / pool_size)
    for i in range(pool_size):
        t_results.append(pool.apply_async(single_thread_gen, args=[data_size_per_thread]))
    pool.close()
    pool.join()

    data = []
    label = []
    for r in t_results:
        for x in r.get():
            data.append(x[0:8])
            label.append(x[8])

    while len(data) > data_size:
        data.remove()
        label.remove()

    t1 = time.time()
    gen_time = t1 - t0
    print("get_data: ", data_size, gen_time)

    return data, label


if __name__ == '__main__':
    size = 1000
    # data, label = multi_thread_gen(data_size=size)
    # data, label = get_data_max_values(data_size=size)
    # data, label = multithreading(data_size=size, pool_size=16)
    data, label = multi_thread_gen(data_size=size, pool_size=10)
    # data, label = multithreading(data_size=size, pool_size=4)
    # data, label = multithreading(data_size=size, pool_size=2)
    # data, label = multithreading(data_size=size, pool_size=1)
    # data, label = get_data_max_values(data_size=size)
    print(len(label))
    1==1


