#%tensorflow_version 2.x
import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
from tensorflow.keras import layers
import random
import itertools
import copy
import os
from tensorflow.keras.models import load_model
import statistics
import time
import socket

from tensorflow.keras.callbacks import Callback
import nn_relay_util as util


def verify(size=10000, verbose=0, model=None):
    config = util.ExpConfig()

    data_size = size

    # construct valuation data
    val_data, val_label = util.get_data_max_values(data_size)

    # load model
    if model is None:
        model_name = "nn_relay_model"

        if os.path.exists(model_name + ".h5"):
            print("load model")
            model = load_model(model_name + ".h5")
        else:
            print("no model found")
            return

    # prediction data
    pre_label = model.predict(val_data)

    # calculate
    wrong = 0
    sum_squared_err = 0
    err = 1 / (config.M * config.K) / 2
    for i in range(data_size):
        w = 0
        for j in range(config.K):
            if abs(val_label[i][j] - pre_label[i][j]) > err:
                w += 1
                if verbose == 1:
                    print(wrong, " ====")
                    print(val_label[i])
                    print(pre_label[i])
                sum_squared_err += 1
        if w > 0:
            wrong += 1
    MSE = sum_squared_err / data_size
    acc = (data_size - wrong) / data_size
    if verbose == 1:
        print("wrong:", wrong)
        print("accuracy:", acc)
        print("MSE", MSE)
    return MSE, acc


def verify_with_outage_prob(size=10000, model=None, verbose=0, export_to_files=False):
    config = util.ExpConfig()

    # load model
    if model is None:
        model_name = "latest"

        if os.path.exists(model_name + ".h5"):
            print("load model")
            model = load_model(model_name + ".h5")
        else:
            print("no model found")
            return

    data_size = size
    data = []
    input = []
    result = []
    data_norm = []
    for i in range(data_size):
        G, gn, gidx = util.generate_G()
        G_idx = util.get_idx(G)
        G_max = G_idx.max()
        G_idx_norm = G_idx / G_max
        data_norm.append(G_idx_norm)
        G_idx_norm_reshape = np.reshape(G_idx_norm, [config.M * config.K])
        data.append(G)
        input.append(G_idx_norm_reshape.tolist())

    # get predict result
    pre_result = model.predict(input)
    for i in range(data_size):
        r = util.get_relay_by_selection(data_norm[i], pre_result[i])
        result.append(r)

    # get brute force result
    bf_result = []
    wrong = 0
    sum_squared_error = 0
    for i in range(data_size):
        G = data[i]
        r_bf = np.array(util.brute_force2(G))
        r_pre = np.array(result[i])
        bf_result.append(r_bf)
        squared_error = np.sum(np.power(r_bf - r_pre, 2))
        sum_squared_error += squared_error
        if squared_error > 0:
            wrong += 1
            if verbose == 1:
                print(r_bf)
                print(result[i])
                print("==========")


    # get random selection result
    random_result = []
    for i in range(data_size):
        r = [0] * config.M
        # randomly pick 1 to L relays
        for k in range(config.L):
            r[random.randint(0, config.M - 1)] = 1
        random_result.append(r)

    # get bulk selection result
    bulk_result = []
    for i in range(data_size):
        dt = np.array(data[i])
        first_min = np.min(dt, axis=1)
        second_max = np.max(first_min)
        select_idx = np.argwhere(dt == second_max)
        r = [0] * config.M
        r[select_idx[0][0]] = 1
        bulk_result.append(r)

    # get per-sub result
    persub_result = []
    for i in range(data_size):
        dt = np.array(data[i])
        r = [0] * config.M
        for k in range(config.L):
            tmp = dt[:, k]
            select_idx = np.argwhere(tmp == max(tmp))
            r[select_idx[0][0]] = 1
        persub_result.append(r)

    # calculate MSE and acc
    acc = (data_size - wrong) / data_size
    MSE = sum_squared_error / data_size

    # outage probability
    out_prob_bf = util.get_outage_prob(data, bf_result)
    out_prob_predict = util.get_outage_prob(data, result)
    op_random = util.get_outage_prob(data, random_result)
    op_bulk = util.get_outage_prob(data, bulk_result)
    op_persub = util.get_outage_prob(data, persub_result)

    # export input matrix G, predicted results and
    # results by brute-force search to file under 'verify' folder
    if export_to_files:
        with open("verify/G.csv", 'w') as f:
            for dt in data:
                for l in range(config.L):
                    for m in range(config.M):
                        f.write(str(dt[m, l])+',')
                    f.write("\n")

        with open("verify/results.csv", 'w') as f:
            # one brute-force searched result
            # followed by one predicted result row
            for i in range(len(pre_result)):
                for r in bf_result[i]:
                    f.write(str(r)+',')
                if bf_result[i].tolist() != result[i]:
                    f.write('not match,')
                f.write('\n')
                for r in result[i]:
                    f.write(str(r)+',')
                f.write('\n')

    return MSE, acc, out_prob_bf, out_prob_predict, op_random, op_bulk, op_persub


def main():
    MSE, acc = verify(verbose=1)
    print(MSE, acc)


def matlab_verify():
    config = util.ExpConfig()

    model_name = "nn_relay_model"
    model = load_model(model_name + ".h5")

    data_size = 10000
    data = []
    input = []
    result = []
    data_norm = []
    for i in range(data_size):
        G, gn, gidx = util.generate_G()
        G_idx = util.get_idx(G)
        G_max = G_idx.max()
        G_idx_norm = G_idx / G_max
        data_norm.append(G_idx_norm)
        G_idx_norm_reshape = np.reshape(G_idx_norm, [config.M * config.K])
        data.append(G)
        input.append(G_idx_norm_reshape.tolist())

    pre_result = model.predict(input)
    for i in range(data_size):
        r = util.get_relay_by_selection(data_norm[i], pre_result[i])
        result.append(r)

    bf_result = []
    wrong = 0
    for i in range(data_size):
        G = data[i]
        r = util.brute_force2(G)
        if r != result[i]:
            wrong += 1
            print(r)
            print(result[i])
            print("==========")
    print("wrong: ", wrong)
    print("acc: ", (data_size - wrong) / data_size)

    # write file to matlab
    f = open("out.txt", "w")
    for i in range(data_size):
        # write input
        for k in range(config.K):
            s = ""
            for m in range(config.M):
                s += str(data[i][m][k])
                if m < config.M - 1:
                    s += ","
            f.write(s + "\n")

        s = ""
        # write output
        for m in range(config.M):
            s += str(result[i][m])
            if m < config.M - 1:
                s += ","
        f.write(s + "\n")


if __name__ == "__main__":
    (
        MSE,
        acc,
        out_prob_bf,
        out_prob_predict,
        op_random,
        op_bulk,
        op_persub,
    ) = verify_with_outage_prob(size=100, export_to_files=True)
    print(MSE, acc)
