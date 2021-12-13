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
    G_idx = np.array(range(1, config.M * config.K + 1))
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
        G_idx_full_reshape = np.reshape(idx_norm, [config.M * config.K])
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


def generate_G(config=None, target=None):
    if config == None:
        config = ExpConfig()
    G1 = np.random.exponential(scale=1, size=[config.M, config.K])
    G2 = np.random.exponential(scale=1, size=[config.M, config.K])
    G = subcarrier_min(G1, G2)
    G_min = G.min()
    G_max = G.max()
    G_normalized = 2 * (G - G_min) / (G_max - G_min) - 1
    G_reshape = np.sort(G.reshape(1, config.M * config.K))
    G_flip = np.flip(G_reshape)
    G_idx_full = copy.deepcopy(G)
    for i in range(config.M):
        for j in range(config.K):
            G_idx_full[i, j] = np.where(G_flip == G_idx_full[i, j])[1]
    # print(G)
    # print("*******************")
    # print(G_normalized)
    # print(G_idx_full)
    return G, G_normalized, G_idx_full / (config.M * config.K)


def subcarrier_min(a, b):
    c = copy.deepcopy(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if c[i, j] > b[i, j]:
                c[i, j] = b[i, j]
    return c


def brute_force2(G, config=None):
    if config is None:
        config = ExpConfig()
    max_values = get_max_values(G)
    S_d_relay = [0] * config.M
    for i in range(config.M):
        if max_values in G[i]:
            S_d_relay[i] += 1
    return S_d_relay


def get_idx(G):
    config = ExpConfig()
    G_idx_full = copy.deepcopy(G)
    G_reshape = np.sort(G.reshape(1, config.M * config.K))
    G_flip = np.flip(G_reshape)
    for i in range(config.M):
        for j in range(config.K):
            G_idx_full[i, j] = np.where(G_reshape == G_idx_full[i, j])[1]
    G_idx_full = G_idx_full + 1
    return G_idx_full


def get_data(config, data_size):
    data = []
    label = []
    t0 = time.time()
    for i in range(round(data_size)):
        G, G_normalized, G_idx_full = generate_G()
        G_idx_full_reshape = np.reshape(G_idx_full, [config.M * config.K])
        data.append(G_idx_full_reshape.tolist())
        bf2 = brute_force2(G_idx_full)
        S_d_relay = bf2
        label.append(S_d_relay)
    t1 = time.time()
    gen_time = t1 - t0
    print("get_data", data_size, gen_time)
    return data, label


if __name__ == "__main__":
    pass
