import datetime
import random
import math

import dgl
import torch
import numpy as np

from config import *


config = Config()


def get_device(index=3):
    return torch.device("cuda:" + str(index) if torch.cuda.is_available() else "cpu")


def show_time():
    time_stamp = '\033[1;31;40m[' + str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ']\033[0m'

    return time_stamp


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mix_collate_fn(batch):
    header_data, payload_data, target = list(zip(*batch))
    header_data = np.array(header_data).flatten()
    header_data = dgl.batch(header_data)
    payload_data = np.array(payload_data).flatten()
    payload_data = dgl.batch(payload_data)
    target = torch.LongTensor(target)

    return header_data, payload_data, target


def get_bytes_from_raw(s):
    rows = s.split('\n')
    for i, row in enumerate(rows):
        rows[i] = row[6: 53].strip()

    bytes_list = []
    for row in rows:
        bytes_list.extend(row.split(' '))

    bytes_list_dec = [int(hex, 16) for hex in bytes_list]

    return bytes_list, bytes_list_dec


def pad_truncate(flow, type, config):
    flow_pad_trunc_length = config.FLOW_PAD_TRUNC_LENGTH
    if type == 'payload':
        byte_pad_trunc_length = config.BYTE_PAD_TRUNC_LENGTH
    elif type == 'header':
        byte_pad_trunc_length = config.HEADER_BYTE_PAD_TRUNC_LENGTH

    if len(flow) > flow_pad_trunc_length:
        flow = flow[:flow_pad_trunc_length]

    for ind, p in enumerate(flow):
        if len(p) > byte_pad_trunc_length:
            flow[ind] = p[:byte_pad_trunc_length]
        elif len(p) < byte_pad_trunc_length:
            p.extend([config.PAD_TRUNC_DIGIT] * (byte_pad_trunc_length - len(p)))
            flow[ind] = p

    if len(flow) < flow_pad_trunc_length:
        flow.extend([[config.PAD_TRUNC_DIGIT] * byte_pad_trunc_length] * (flow_pad_trunc_length - len(flow)))

    return flow


def remove(flow):
    for ind, p in enumerate(flow):
        ip_header = p[:20]
        tcp_udp_header = p[20:]
        ip_header = ip_header[:12]
        tcp_udp_header = tcp_udp_header[4:]

        renew_header = []
        renew_header.extend(ip_header)
        renew_header.extend(tcp_udp_header)
        flow[ind] = renew_header

    return flow


def split_flow_ISCX(file_path, cate, allow_empty, pad_trunc, config, type='payload'):
    file = np.load(file_path, allow_pickle=True)
    packets = file[type]
    if type == 'header':
        baseline = file['payload']
    data_list = []

    seg_pcap = packets
    if type == 'header':
        seg_baseline = baseline
    if allow_empty:
        seg_pcap = [list(p) for ind, p in enumerate(seg_pcap)]
        if type == 'header':
            seg_baseline = [list(p) for ind, p in enumerate(seg_baseline)]
    else:
        seg_pcap = [list(p) for ind, p in enumerate(seg_pcap) if len(p) != 0]
        if type == 'header':
            seg_baseline =[list(p) for ind, p in enumerate(seg_baseline) if len(p) != 0]
    if type == 'header':
        if len(seg_baseline) == 0:
            print("Empty Flow Detected")
            return data_list
    else:
        if len(seg_pcap) == 0:
            print("Empty Flow Detected")
            return data_list
    if pad_trunc:
        if type == 'header':
            if len(seg_baseline) > config.ANOMALOUS_FLOW_THRESHOLD:
                print("Anomalous Flow Detected")
                return data_list
            seg_pcap = remove(flow=seg_pcap)
        else:
            if len(seg_pcap) > config.ANOMALOUS_FLOW_THRESHOLD:
                print("Anomalous Flow Detected")
                return data_list
        seg_pcap = pad_truncate(flow=seg_pcap, type=type, config=config)
    data_list.append(seg_pcap)

    return data_list


def split_flow_Tor_nonoverlapping(file_path, cate, allow_empty, pad_trunc, config, type='payload'):
    file = np.load(file_path, allow_pickle=True)
    packets = file[type]
    if type == 'header':
        baseline = file['payload']
    data_list = []
    time_stamp = np.array(file['time']).astype(np.float64)
    time_stamp = time_stamp - time_stamp[0]
    sliding_window = int((time_stamp[-1] - 60) / 60) + 1
    if time_stamp[-1] <= 60:
        sliding_window = 1
    begin = [60 * i for i in range(sliding_window)]
    end = [60 + 60 * i for i in range(sliding_window)]
    all_seg_stamp = list(set(begin + end))
    all_seg_stamp.sort()
    stamp_ind_map = dict()
    prev_j = 0
    for i, seg_stamp in enumerate(all_seg_stamp):
        for j in range(prev_j, len(time_stamp)):
            if seg_stamp <= time_stamp[j]:
                stamp_ind_map[seg_stamp] = j
                prev_j = j
                break
    if time_stamp[-1] <= 60:
        stamp_ind_map[60] = len(time_stamp)
    begin = [stamp_ind_map[i] for i in begin]
    end = [stamp_ind_map[i] for i in end]
    for s_ind, e_ind in zip(begin, end):
        if s_ind == e_ind:
            continue
        seg_pcap = packets[s_ind: e_ind]
        if type == 'header':
            seg_baseline = baseline[s_ind: e_ind]
        if allow_empty:
            seg_pcap = [list(p) for ind, p in enumerate(seg_pcap)]
            if type == 'header':
                seg_baseline = [list(p) for ind, p in enumerate(seg_baseline)]
        else:
            seg_pcap = [list(p) for ind, p in enumerate(seg_pcap) if len(p) != 0]
            if type == 'header':
                seg_baseline =[list(p) for ind, p in enumerate(seg_baseline) if len(p) != 0]
        if type == 'header':
            if len(seg_baseline) == 0:
                print("Empty Flow Detected")
                continue
        else:
            if len(seg_pcap) == 0:
                print("Empty Flow Detected")
                continue
        if pad_trunc:
            if type == 'header':
                if len(seg_baseline) > config.ANOMALOUS_FLOW_THRESHOLD:
                    print("Anomalous Flow Detected")
                    continue
                seg_pcap = remove(flow=seg_pcap)
            else:
                if len(seg_pcap) > config.ANOMALOUS_FLOW_THRESHOLD:
                    print("Anomalous Flow Detected")
                    continue
            seg_pcap = pad_truncate(flow=seg_pcap, type=type, config=config)
        data_list.append(seg_pcap)

    return data_list


def construct_graph(bytes, w_size, k=1):
    # word co-occurence with context windows
    window_size = w_size
    windows = [] # [[], [], [], ..., []]

    words = bytes # ['A', 'B', 'C']
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)


    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])


    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_i
                word_j = window[j]
                word_j_id = word_j
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    src = []
    dst = []
    weight = []

    # pmi as weights

    num_window = len(windows)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[i]
        word_freq_j = word_window_freq[j]
        pmi = math.log((1.0 * count / num_window) ** k /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        src.append(i)
        dst.append(j)
        weight.append(pmi)

    bytes2id = {}
    feat = []
    id_count = 0
    for byte in src:
        if byte in bytes2id:
            continue
        bytes2id[byte] = id_count
        id_count += 1
        feat.append([byte])

    src = [bytes2id[i] for i in src]
    dst = [bytes2id[i] for i in dst]

    g = dgl.graph((src, dst))
    g.ndata['feat'] = torch.tensor(feat, dtype=torch.float32)

    return dgl.add_self_loop(g)


if __name__ == '__main__':
    pass