import os
import pickle
import argparse
import pandas as pd
import numpy as np
from collections import OrderedDict, defaultdict
import math
import random
import json
from random import choice, sample

parser = argparse.ArgumentParser()

parser.add_argument("--train_anomaly_ratio", default=0.15, type=float)

params = vars(parser.parse_args())

eval_name = f'Spirit_{params["train_anomaly_ratio"]}_tar'
seed = 42
data_dir = "./"
np.random.seed(seed)

params = {
    "log_file": "trainsets/Spirit.csv",
    "time_range": 100,  # 21600,  # 6 hours
    "train_ratio": None,
    "test_ratio": 1,
    "random_sessions": False,
    "train_anomaly_ratio": params["train_anomaly_ratio"],
}

data_dir = os.path.join(data_dir, eval_name)
os.makedirs(data_dir, exist_ok=True)


def decision(probability):
    return random.random() < probability


def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(
            obj,
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


def load_BGL(
        log_file,
        time_range,
        train_ratio,
        test_ratio,
        random_sessions,
        train_anomaly_ratio,
):
    print("Loading BGL logs from {}.".format(log_file))
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    # struct_log.sort_values(by=["Timestamp"], inplace=True)

    struct_log["Label"] = struct_log["Label"].map(lambda x: x != "-").astype(int).values

    # struct_log["seconds_since"] = (
    #     (struct_log["Time"] - struct_log["Time"][0]).astype(int)
    # )
    struct_log["seconds_since"] = [i for i in range(len(struct_log))]
    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
    for idx, row in enumerate(struct_log.values):
        current = row[column_idx["seconds_since"]]
        if idx == 0:
            sessid = current
        elif current - sessid > time_range:
            sessid = current
        if sessid not in session_dict:
            session_dict[sessid] = defaultdict(list)
        session_dict[sessid]["templates"].append(row[column_idx["EventTemplate"]])
        session_dict[sessid]["label"].append(
            row[column_idx["Label"]]
        )  # labeling for each log

    # labeling for each session
    # for k, v in session_dict.items():
    #     session_dict[k]["label"] = [int(1 in v["label"])]

    session_idx = list(range(len(session_dict)))
    # split data
    if random_sessions:
        print("Using random partition.")
        np.random.shuffle(session_idx)

    session_ids = np.array(list(session_dict.keys()))
    ###################################################################
    if train_ratio is None:
        train_ratio = 1 - test_ratio
    train_lines = math.ceil(train_ratio * len(session_idx))
    test_lines = len(session_idx) - train_lines
    print('train_lines', train_lines)
    session_idx_train = session_idx[0:train_lines]
    session_idx_test = session_idx[-test_lines:]

    session_id_train = session_ids[session_idx_train]
    session_id_test = session_ids[session_idx_test]

    print("Total # sessions: {}".format(len(session_ids)))
    session_train = {
        k: session_dict[k]
        for k in session_id_train
        if (sum(session_dict[k]["label"]) == 0)
           or (sum(session_dict[k]["label"]) > 0 and decision(train_anomaly_ratio))
    }
    session_test = {k: session_dict[k] for k in session_id_test}
    # 若无法正常运行则启用以下代码
    # if train_anomaly_ratio == 0:
    #     extra_white_samples = {}
    #     # 创建测试集ID的可变副本
    #     test_ids = list(session_id_test.copy())
    #     index = 0
    #     needed = len(session_id_train) - len(session_train)
    #
    #     while needed > 0 and index < len(test_ids):
    #         sess_id = test_ids[index]
    #         current_session = session_test[sess_id]
    #
    #         if sum(current_session["label"]) == 0:
    #             extra_white_samples[sess_id] = current_session
    #             del session_test[sess_id]  # 从测试集删除
    #             test_ids.pop(index)  # 从ID列表删除
    #             needed -= 1
    #         else:
    #             index += 1  # 只有未删除时才增加索引
    #
    #     session_train.update(extra_white_samples)
    if train_anomaly_ratio == 0:
        extra_white_samples = {}
        index = 0
        while len(extra_white_samples) < (len(session_id_train) - len(session_train)):
            current_session = session_test[session_id_test[index]]
            if sum(current_session["label"]) == 0:
                extra_white_samples[session_id_test[index]] = current_session
                del session_test[session_id_test[index]]
            index += 1
        session_train.update(extra_white_samples)
    ###################################################################
    # black_ids=[k for k in session_ids if sum(session_dict[k]["label"]) > 0]
    # white_ids=[k for k in session_ids if sum(session_dict[k]["label"]) == 0]
    # if train_anomaly_ratio==0:
    #     session_id_train=sample(white_ids,2)
    # else:
    #     session_id_train=[choice(black_ids),choice(white_ids)]
    # session_id_test=[k for k in session_ids if k not in session_id_train]
    # session_train = {k: session_dict[k] for k in session_id_train}
    # session_test = {k: session_dict[k] for k in session_id_test}
    ###################################################################
    session_labels_train = [
        1 if sum(v["label"]) > 0 else 0 for _, v in session_train.items()
    ]
    session_labels_test = [
        1 if sum(v["label"]) > 0 else 0 for _, v in session_test.items()
    ]

    if len(session_labels_train) != 0:
        train_anomaly = 100 * sum(session_labels_train) / len(session_labels_train)
        print("# train sessions: {} ({:.2f}%)".format(len(session_train), train_anomaly))
    if len(session_labels_test) != 0:
        test_anomaly = 100 * sum(session_labels_test) / len(session_labels_test)
        print("# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly))


    with open(os.path.join(data_dir, "session_train.pkl"), "wb") as fw:
        pickle.dump(session_train, fw)
    with open(os.path.join(data_dir, "session_test.pkl"), "wb") as fw:
        pickle.dump(session_test, fw)
    json_pretty_dump(params, os.path.join(data_dir, "data_desc.json"))
    print("Saved to {}".format(data_dir))
    return session_train, session_test


if __name__ == "__main__":
    load_BGL(**params)