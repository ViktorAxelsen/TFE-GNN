import os
import argparse
import torch
import dgl
import numpy as np

from utils import show_time, construct_graph, split_flow_Tor_nonoverlapping, split_flow_ISCX
from config import *


generalConfig = Config()


def construct_dataset_from_bytes_ISCX(dir_path_dict, type):
    train = []
    train_label = []
    test = []
    test_label = []
    TRAIN_FLOW_COUNT = dict()
    TEST_FLOW_COUNT = dict()
    for category in dir_path_dict:
        dir_path = dir_path_dict[category]
        file_list = os.listdir(dir_path)
        data_list = []
        for file in file_list:
            if not file.endswith('.npz'):
                continue
            file_path = dir_path + '/' + file
            print('{} {} Process Starting'.format(show_time(), file_path))
            if opt.dataset == 'iscx-tor':
                data_list.extend(split_flow_Tor_nonoverlapping(file_path, category, allow_empty=False, pad_trunc=True, config=config, type=type))
            else:
                data_list.extend(split_flow_ISCX(file_path, category, allow_empty=False, pad_trunc=True, config=config, type=type))

        data_list = data_list[:config.MAX_SEG_PER_CLASS]
        split_ind = int(len(data_list) / 10)
        data_list_train = data_list[split_ind + 1:]
        data_list_test = data_list[: split_ind + 1]

        train.extend(data_list_train)
        train_label.extend([category] * len(data_list_train))
        test.extend(data_list_test)
        test_label.extend([category] * len(data_list_test))

        TRAIN_FLOW_COUNT[category] = len(data_list_train)
        TEST_FLOW_COUNT[category] = len(data_list_test)
        print(TRAIN_FLOW_COUNT[category], TEST_FLOW_COUNT[category])

    if type == 'payload':
        np.savez_compressed(config.TRAIN_DATA, data=np.array(train), label=np.array(train_label))
        np.savez_compressed(config.TEST_DATA, data=np.array(test), label=np.array(test_label))
    elif type == 'header':
        np.savez_compressed(config.HEADER_TRAIN_DATA, data=np.array(train), label=np.array(train_label))
        np.savez_compressed(config.HEADER_TEST_DATA, data=np.array(test), label=np.array(test_label))

    print(TRAIN_FLOW_COUNT)
    print(TEST_FLOW_COUNT)


def construct_graph_format_data(file_path, save_path, type, w_size=generalConfig.PMI_WINDOW_SIZE, pmi=1):
    file = np.load(file_path, allow_pickle=True)
    gs = []
    if type == 'payload':
        data = file['data'].reshape(-1, config.BYTE_PAD_TRUNC_LENGTH)
    elif type == 'header':
        data = file['data'].reshape(-1, config.HEADER_BYTE_PAD_TRUNC_LENGTH)
    label = file['label']
    for ind, p in enumerate(data):
        gs.append(construct_graph(bytes=p, w_size=w_size, k=pmi))
        if ind % 500 == 0:
            print('{} {} Graphs Constructed'.format(show_time(), ind))

    dgl.save_graphs(save_path, gs, {"glabel": torch.LongTensor(label)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    opt = parser.parse_args()

    if opt.dataset == 'iscx-vpn':
        config = ISCXVPNConfig()
    elif opt.dataset == 'iscx-nonvpn':
        config = ISCXNonVPNConfig()
    elif opt.dataset == 'iscx-tor':
        config = ISCXTorConfig()
    elif opt.dataset == 'iscx-nontor':
        config = ISCXNonTorConfig()
    else:
        raise Exception('Dataset Error')

    construct_dataset_from_bytes_ISCX(dir_path_dict=config.DIR_PATH_DICT, type='payload')
    construct_graph_format_data(file_path=config.TRAIN_DATA, save_path=config.TRAIN_GRAPH_DATA, type='payload')
    construct_graph_format_data(file_path=config.TEST_DATA, save_path=config.TEST_GRAPH_DATA, type='payload')
    construct_dataset_from_bytes_ISCX(dir_path_dict=config.DIR_PATH_DICT, type='header')
    construct_graph_format_data(file_path=config.HEADER_TRAIN_DATA, save_path=config.HEADER_TRAIN_GRAPH_DATA, type='header')
    construct_graph_format_data(file_path=config.HEADER_TEST_DATA, save_path=config.HEADER_TEST_GRAPH_DATA, type='header')