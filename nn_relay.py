# system
import argparse
import sys
import os
import copy
import time
from datetime import datetime

# calculations
import statistics

# tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow_core.python.keras.callbacks import Callback

# plotting
import matplotlib

# supports
import nn_relay_util as util
import nn_relay_verify as v
from nn_relay_util import ExpConfig as config

# train record related
from shutil import copyfile
import pickle


def new_model():
    # get experiment configurations
    M = config.M
    K = config.K
    model = tf.keras.Sequential([
        layers.Dense(M * K, activation='relu', input_shape=(M * K,)),
        layers.Dense(M * M * K, activation='relu', kernel_initializer=keras.initializers.glorot_normal()),
        layers.Dense(M * M * K * 2, activation='relu', kernel_initializer=keras.initializers.glorot_normal()),
        layers.Dense(M * M * K * 4, activation='relu', kernel_initializer=keras.initializers.glorot_normal()),
        layers.Dense(M * M * K * 8, activation='relu', kernel_initializer=keras.initializers.glorot_normal()),
        # layers.Dense(M * M * K * 16, activation='relu', kernel_initializer=keras.initializers.glorot_normal()),
        layers.Dense(M * M * K * 8, activation='relu', kernel_initializer=keras.initializers.glorot_normal()),
        layers.Dense(M * M * K * 4, activation='relu', kernel_initializer=keras.initializers.glorot_normal()),
        layers.Dense(M * M * K * 2, activation='relu', kernel_initializer=keras.initializers.glorot_normal()),
        layers.Dropout(rate=0.1),
        layers.Dense(K, kernel_initializer=keras.initializers.glorot_normal())
    ])

    # compile model
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.mean_squared_error,
                  metrics=['accuracy'])
    return model


class ProgramConfig:
    model_name = "NOT_SET"
    main_path = "records/NOT_SET"
    lr = 0.001
    lr_adaptive = True
    lr_base = 0.001
    backup_filenames = ["nn_local.py", "nn_relay_util.py", "nn_relay_verify.py"]
    target_acc = 0.984


class ProgressCallback(Callback):
    def __init__(self, ss=None):
        # how many epochs between '#' display on terminal
        self.step = 20
        self.c = self.step

    def on_train_begin(self, logs=None):
        print("train_begin")

    def on_epoch_end(self, epoch, logs=None):
        self.c -= 1
        if self.c <= 0:
            self.c = self.step
            sys.stdout.write('#')
            sys.stdout.flush()

    def on_train_end(self, logs=None):
        print("train_end")


def write_log(s0, file="log_main", print_msg=False):
    t = datetime.now().strftime("%Y%m%d %H:%M:%S ")
    s = t + s0
    with open("records/"+ProgramConfig.model_name+"/logs/"+file+".txt", 'a+') as f:
        f.write(s + "\n")
    if print_msg:
        print(s)


def new_training():
    print("creating new training directories")
    main_path = "records/" + ProgramConfig.model_name
    ProgramConfig.main_path = main_path
    os.makedirs(main_path, exist_ok=True)
    os.makedirs(main_path + "/models", exist_ok=True)
    os.makedirs(main_path + "/his_dump", exist_ok=True)
    os.makedirs(main_path + "/logs", exist_ok=True)
    os.makedirs(main_path + "/src_bak", exist_ok=True)

    # backup src files
    for s in ProgramConfig.backup_filenames:
        copyfile(s, main_path + "/src_bak/" + s)

    # create model
    model = new_model()

    return model


def nn_local(load=None):
    # ========= load or create model =============
    if load is None:
        # start a brand new training
        # training data would be store in a directory named by launch time e.g. 20200413_1030
        d = datetime.now()
        ProgramConfig.model_name = d.strftime("%Y%m%d_%H%M")
        model = new_training()
        his_sum = None
        start_round = 0
        print("init learning rate: ", ProgramConfig.lr)
    else:
        # load and continues a previous training
        # directory name must provide by parameter [load]
        # previous training directories would be under ../records/
        ProgramConfig.model_name = load
        ProgramConfig.main_path = "records/"+ProgramConfig.model_name
        print("load model")
        model = load_model(ProgramConfig.main_path + "/models/latest.h5")
        # load his_sum
        with open(ProgramConfig.main_path + "/his_dump/sum.hisdump", "rb") as f:
            his_sum = pickle.load(f)
            start_round = his_sum["round_num"] + 1
            ProgramConfig.lr = his_sum["lr"]
            mse_non_improve = his_sum["mse_non_improve"]
            best_mse = his_sum["best_mse"]

    # ============= training =============
    # init
    mse_non_improve = 0
    best_mse = -1
    max_rounds = 200
    train_data = None
    lr_patience = 20

    # enter training rounds
    for round_num in range(start_round, max_rounds):
        epochs = 1000
        # init callback functions
        early_stop = keras.callbacks.EarlyStopping(patience=30, monitor='loss', restore_best_weights=False)
        # this is a custom callback function for displaying training progress on terminal
        prog_back = ProgressCallback()

        # total number of entire data size in this training round
        data_size = 500

        # get the training and validation data set
        train_data, train_label = util.get_data_max_values(data_size)
        val_data, val_label = util.get_data_max_values(data_size / 10)

        # a timer to keep track on training time
        t0 = time.time()

        # train the NN and record history
        his = model.fit(x=train_data, y=train_label, epochs=epochs, batch_size=round(data_size / 4),
                        validation_data=(val_data, val_label), callbacks=[early_stop, prog_back], verbose=0)
        print("train_time", time.time() - t0)

        loss = statistics.mean(his.history['loss'])
        val_loss = statistics.mean(his.history['val_loss'])

        # launch verify function to get the MSE and acc from the aspect of relay selections
        # (rather than MSE of "G after selection")
        MSE, acc, out_prob_bf, out_prob_predict, op_random, op_bulk, op_persub = v.verify_with_outage_prob(model=model)
        verify_data = [MSE, acc, out_prob_bf, out_prob_predict, op_random, op_bulk, op_persub]

        # report progress to terminal and log
        round_report = "Round-" + str(round_num) + " MSE:" + str(MSE) + " acc:" + str(acc) + " loss:" + \
                       str(loss) + " val_loss:" + str(val_loss) + \
                       "\nout_prob_bf:"+str(out_prob_bf)+" out_pro_pre:"+str(out_prob_predict)
        print(round_report)
        write_log(round_report)

        # record model and training history data
        print("recording...")

        # save the model and overwrite the latest version
        # model.save(ProgramConfig.main_path + "/models/model_r" + str(round_num) + ".h5")
        model.save(ProgramConfig.main_path + "/models/latest.h5")

        # dump History object of this round
        his_dic = {}
        tmp_keywords = ["loss", "val_loss", "accuracy", "val_accuracy"]
        his_dic["params"] = []
        his_dic["verify"] = verify_data
        for k in tmp_keywords:
            his_dic[k] = copy.deepcopy(his.history[k])
        his_dic["params"].append(copy.deepcopy(his.params))
        with open(ProgramConfig.main_path + "/his_dump/r"+str(round_num)+".hisdump", "wb") as f:
            pickle.dump(his_dic, f)

        # accumulating total training history
        # with current program settings
        if his_sum is None:
            his_sum = {}
            for k in tmp_keywords:
                his_sum[k] = []
            his_sum["params"] = []
            his_sum["verify"] = []
        for k in tmp_keywords:
            his_sum[k] += his.history[k]
        his_sum["params"].append(his_dic["params"][0])
        his_sum["round_num"] = round_num
        his_sum["verify"].append(verify_data)
        his_sum["lr"] = ProgramConfig.lr
        his_sum["mse_non_improve"] = mse_non_improve
        his_sum["best_mse"] = best_mse
        with open(ProgramConfig.main_path + "/his_dump/sum.hisdump", "wb") as f:
            pickle.dump(his_sum, f)

        # plain text history
        pt_his = ""
        for i in range(len(his.history['loss'])):
            pt_his += str(his.epoch[i]) + "," + str(his.history['loss'][i])+","+str(his.history['accuracy'][i])\
                      +","+str(his.history['val_loss'][i])+","+str(his.history['val_accuracy'][i])+"\n"
        with open(ProgramConfig.main_path + "/logs/his_log_r" + str(round_num) + ".txt", "w") as f:
            f.write(pt_his)

        print("history saved")

        # check if reach target acc
        if acc > ProgramConfig.target_acc:
            msg = "target accuracy reached ("+str(acc)+"/"+str(ProgramConfig.target_acc)+"), training terminated"
            write_log(msg, print_msg=True)
            return

        # adaptive learning rate:
        # if mse not improve for [lr_patience] rounds
        # try to lower learning rate by 10 times
        if best_mse == -1 or loss < best_mse:
            best_mse = loss
            mse_non_improve = 0
        else:
            mse_non_improve += 1
        if mse_non_improve >= lr_patience and ProgramConfig.lr > 0.1**2 * ProgramConfig.lr_base:
            ProgramConfig.lr *= 0.1
            mse_non_improve = 0
            best_mse = -1
            print("==========================================")
            print("learning rate change: ", ProgramConfig.lr)
            write_log("learning rate change: "+str(ProgramConfig.lr))
        if ProgramConfig.lr_adaptive:
            keras.backend.set_value(model.optimizer.lr, ProgramConfig.lr)
        else:
            print("lr_adaptive = False, discard lr change")

        # lr suppress:
        # acc can be higher than 0.9 even with initial learning rate (0.1**3)
        # however this learning rate is too high to keep the acc level hence lead to an "overshoot" by the optimizer
        # here would lower the learning rate if it's still higher than 0.1**4 when acc meets 0.9
        if acc > 0.9 and ProgramConfig.lr > 0.1 * ProgramConfig.lr_base:
            ProgramConfig.lr = 0.1 * ProgramConfig.lr_base
            mse_non_improve = 0
            print("==========================================")
            write_log("lr suppress: "+str(ProgramConfig.lr), print_msg=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-n", help="train a brand new model", action="store_true")
    group.add_argument("-l", help="continue training existing model, -l [model_name]", type=str)
    args = parser.parse_args()
    if not args.n and args.l is None:
        print("err: no run mode assigned, -h for more info")
    elif args.n:
        nn_local()
    else:
        nn_local(args.l)

# git test...
