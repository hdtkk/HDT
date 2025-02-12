import numpy as np
import pandas as pd
import random
import torch

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset


#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Available dataset: {list(dataset_recipes.keys())}")
# traffic_dataset = get_dataset("traffic_nips", regenerate=False)
# ele_dataset = get_dataset("electricity_nips", regenerate=False)



# test_fun = lambda x: 0.0
# batch_size1 = 963
# batch_size2 = 370

def mix_test_dataset(list1, list2, batch_size1=963, batch_size2=370):
    list3 = []
    start_index1 = 0
    start_index2 = 0
    # 循环添加元素到 list3，直到 list1 和 list2 中的元素全部添加完
    while start_index1 < len(list1) or start_index2 < len(list2):
        # 添加 list1 中的元素
        for i in range(start_index1, min(start_index1 + batch_size1, len(list1))):
            list3.append(list1[i])

        # 添加 list2 中的元素
        for i in range(start_index2, min(start_index2 + batch_size2, len(list2))):
            list3.append(list2[i])

        # 更新 list1 和 list2 的起始索引
        start_index1 += batch_size1
        start_index2 += batch_size2
    # print(len(list3))
    return list3


def find_timestamp(dataset, first_timestamp=None, last_timestamp=None):
    for data in dataset:
        timestamp = data['start']
        timestamp = timestamp.asfreq('D')
        if first_timestamp is None:
            first_timestamp = timestamp

        if last_timestamp is None:
            last_timestamp = timestamp
        first_timestamp = min(first_timestamp, timestamp)
        last_timestamp = max(
            last_timestamp,
            timestamp + len(data['target']) - 1,
        )
    return first_timestamp, last_timestamp


def change_freq(a, c):
    for ci, ai in zip(c, a):
        start = pd.Period("2006-01-01", freq="D")
        ci['start'] = start
        ai['start'] = start
    return a, c

def test_change_freq(b, d):
    for bi, di in zip(b, d):
        start = pd.Period("2008-01-02", freq="D")
        bi['start'] = start
        di['start'] = start
    return b, d



def get_train_data(dataset, first_timestamp, last_timestamp=None):
    train_list = []
    for data in dataset:
        data_series = pd.Series(
            data['target'],
            index=pd.period_range(
                start= data['start'].asfreq('D'),
                periods=len(data['target']),
                freq= first_timestamp.freq,
            ),
        )
        full_data = data_series.reindex(
            pd.period_range(
                start=first_timestamp,
                end=last_timestamp,
                freq=first_timestamp.freq,
            ),
            fill_value= np.mean(data_series),
        ).values
        target = full_data
        data = {
            "start": data['start'].asfreq('D'),
            "target": target,
            "feat_static_cat": data[
                "feat_static_cat"
            ].copy(),
            "item_id": data['item_id'],
        }
        train_list.append(data)
    return ListDataset(data_iter=train_list, freq=train_list[0]['start'].freq, one_dim_target=True)



def scale_dataset(dataset_list):
    # gluonts_ds_scaled = []
    scaled_list = []
    # print(dataset_list)
    for data in dataset_list:
        lmax = data['target'].max()
        lmin = data['target'].min()
        if lmax == 0.0:
            lmax += 1e-5
            print('lmax', lmax)
            print('lmin', lmin)
        scale_data = (data['target'] - lmin) / (lmax - lmin)
        d = {
            "target": scale_data,
            "start": data['start'],
            "feat_static_cat": data[
            "feat_static_cat"
                    ].copy(),
            "item_id": data['item_id'],
                }
        # print(d)
        scaled_list.append(d)
    return ListDataset(data_iter=scaled_list, freq=scaled_list[0]['start'].freq, one_dim_target=True)

    #     transformed_dataset = list(iter(i))
    #     scaled_list = []
    #     # for data in transformed_dataset:
    #     #     print('data', data)
    #     #     dasd
    #         lmax = data['target'].max()
    #         lmin = data['target'].min()
    #     #     if lmax == 0.0:
    #     #         lmax += 1e-5
    #     #         print('lmax', lmax)
    #     #         print('lmin', lmin)
    #     #     scale_data = (data['target'] - lmin) / (lmax - lmin)
    #     #     d = {
    #     #         "target": scale_data,
    #     #         "start": data['start'],
    #     #         "feat_static_cat": data[
    #     #             "feat_static_cat"
    #     #         ].copy(),
    #     #         "item_id": data['item_id'],
    #     #     }
    #     #     c
    #     #     # one_dim_target=True
    #     gluonts_ds_scaled.append(ListDataset(data_iter=scaled_list, freq=scaled_list[0]['start'].freq, one_dim_target=True))
    # return gluonts_ds_scaled



def get_test_data(dataset, first_timestamp, last_timestamp=None):
    test_list = []
    for data in dataset:
        data_series = pd.Series(
            data['target'],
            index=pd.period_range(
                start=data['start'].asfreq('D'),
                periods=len(data['target']),
                freq=first_timestamp.freq,
            ),
        )
        full_data = data_series.reindex(
            pd.period_range(
                start=first_timestamp,
                end=data_series.index[-1],
                freq=first_timestamp.freq,
            ),
            fill_value=test_fun(data_series),
        ).values
        # print(type(full_data))
        # dsad
        target = full_data
        data = {
            "start": first_timestamp,
            "target": target,
            "feat_static_cat": data[
                "feat_static_cat"
            ].copy(),
            "item_id": data['item_id'],
        }
        test_list.append(data)
    return ListDataset(data_iter=test_list, freq=test_list[0]['start'].freq, one_dim_target=True)


def get_scale(train_data):
    train_min_list = []
    train_max_list = []

    for i in train_data:
        lmax = i['target'].max()
        lmin = i['target'].min()
        if int(lmax) == 0:
            lmax += 1e-5
        train_min_list.append(lmin)
        train_max_list.append(lmax)
    max_scale = torch.tensor(train_max_list, dtype=torch.float).unsqueeze(0)
    min_scale = torch.tensor(train_min_list, dtype=torch.float).unsqueeze(0)
    data_scale = [max_scale, min_scale]
    return data_scale



