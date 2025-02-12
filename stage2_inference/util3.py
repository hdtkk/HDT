import numpy as np
import pandas as pd
import random
import torch

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset




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



def compute_PICP(batch_true_y, batch_pred_y, cp_range=(2.5, 97.5), return_CI=False):
    """
    Another coverage metric.
    """
    # batch_true_y = batch_true_y.squeeze()
    # batch_pred_y = batch_pred_y.squeeze()
    # batch_true_y = batch_true_y.reshape(-1,1)
    # batch_pred_y = batch_pred_y.reshape(-1,batch_pred_y.shape[-1])
    low, high = cp_range
    CI_y_pred = np.percentile(batch_pred_y, q=[low, high], axis=1)
    y_in_range = (batch_true_y >= CI_y_pred[0]) & (batch_true_y <= CI_y_pred[1])
    coverage = y_in_range.mean()
    if return_CI:
        return coverage, CI_y_pred, low, high
    else:
        return coverage, low, high

def qice(batch_true_y, batch_pred_y, n_bins=10, verbose=False):
    quantile_list = np.arange(n_bins + 1) * (100 / n_bins)
    # batch_true_y = batch_true_y.squeeze()
    # batch_pred_y = batch_pred_y.squeeze()
    batch_true_y = batch_true_y.reshape(-1,1)
   # print(batch_true_y.shape)
    batch_pred_y = batch_pred_y.reshape(-1,batch_pred_y.shape[-1])
    #print(batch_pred_y.shape)
    # compute generated y quantiles
    y_pred_quantiles = np.percentile(batch_pred_y, q=quantile_list, axis=1)
    y_true = batch_true_y.T
    quantile_membership_array = ((y_true - y_pred_quantiles) > 0).astype(int)
    y_true_quantile_membership = quantile_membership_array.sum(axis=0)
    # y_true_quantile_bin_count = np.bincount(y_true_quantile_membership)
    y_true_quantile_bin_count = np.array(
        [(y_true_quantile_membership == v).sum() for v in np.arange(n_bins + 2)])
    #print(y_pred_quantiles.shape, quantile_membership_array.shape, y_true_quantile_bin_count.shape)
    if verbose:
        y_true_below_0, y_true_above_100 = y_true_quantile_bin_count[0], \
                                            y_true_quantile_bin_count[-1]
        print(("We have {} true y smaller than min of generated y, " + \
                      "and {} greater than max of generated y.").format(y_true_below_0, y_true_above_100))
    # combine true y falls outside of 0-100 gen y quantile to the first and last interval
    y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
    y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
    y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]
    # compute true y coverage ratio for each gen y quantile interval
   # print('batch_true_y', batch_true_y)
    y_true_ratio_by_bin = y_true_quantile_bin_count_ / len(batch_true_y)
    # print(y_true_ratio_by_bin)
    assert np.abs(
        np.sum(y_true_ratio_by_bin) - 1) < 1e-10, "Sum of quantile coverage ratios shall be 1!"
    qice_coverage_ratio = np.absolute(np.ones(n_bins) / n_bins - y_true_ratio_by_bin).mean()
    return qice_coverage_ratio, y_true_ratio_by_bin


