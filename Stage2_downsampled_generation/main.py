import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from util3 import *
import torch

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
# from pts.model.tempflow import TempFlowEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator




# Device setting
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Available dataset: {list(dataset_recipes.keys())}")


# gluonts datasets setting
dataset = get_dataset("taxi_30min", regenerate=False)
# print(dataset.metadata.feat_static_cat[0].cardinality)

train_grouper = MultivariateGrouper(max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))
test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)),
                                   max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))

# scaling datasets (train/test)
scaled_train_data = scale_dataset(dataset.train)
scaled_test_data = scale_dataset(dataset.test)


dataset_train = train_grouper(scaled_train_data)
dataset_test = test_grouper(scaled_test_data)


# eevaluator setting
evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                  target_agg_funcs={'sum': np.sum})

# draw the visualizations
def plot(target, forecast, prediction_length, prediction_intervals=(50.0, 90.0), color='g', fname=None):
    label_prefix = ""
    rows = 4
    cols = 4
    fig, axs = plt.subplots(rows, cols, figsize=(24, 24))
    axx = axs.ravel()
    seq_len, target_dim = target.shape

    ps = [50.0] + [
        50.0 + f * c / 2.0 for c in prediction_intervals for f in [-1.0, +1.0]
    ]

    percentiles_sorted = sorted(set(ps))

    def alpha_for_percentile(p):
        return (p / 100.0) ** 0.3

    for dim in range(0, min(rows * cols, target_dim)):
        ax = axx[dim]

        target[-2 * prediction_length:][dim].plot(ax=ax)

        ps_data = [forecast.quantile(p / 100.0)[:, dim] for p in percentiles_sorted]
        i_p50 = len(percentiles_sorted) // 2

        p50_data = ps_data[i_p50]
        p50_series = pd.Series(data=p50_data, index=forecast.index)
        p50_series.plot(color=color, ls="-", label=f"{label_prefix}median", ax=ax)

        for i in range(len(percentiles_sorted) // 2):
            ptile = percentiles_sorted[i]
            alpha = alpha_for_percentile(ptile)
            ax.fill_between(
                forecast.index,
                ps_data[i],
                ps_data[-i - 1],
                facecolor=color,
                alpha=alpha,
                interpolate=True,
            )
            # Hack to create labels for the error intervals.
            # Doesn't actually plot anything, because we only pass a single data point
            pd.Series(data=p50_data[:1], index=forecast.index[:1]).plot(
                color=color,
                alpha=alpha,
                linewidth=10,
                label=f"{label_prefix}{100 - ptile * 2}%",
                ax=ax,
            )

    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    axx[0].legend(legend, loc="upper left")

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.05)


# Taxi as example for testing
# final_state_dict = torch.load('Taxi_96_final_state', map_location='cpu')
stage1_downsampled_state_dict = torch.load('./Taxi_downsampled96_stage1.pkl', map_location='cpu')

estimator = TransformerTempFlowEstimator(
    d_model=256,
    codebook_num=256,
    codebook_beta=0.25,
    latent_dim=256,
    num_heads=8,
    e_layers=3,
    d_layers=2,
    # scaled=train_scale,
    num_parallel_samples=100,
    input_size=1218,
    target_embed_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    factors=20,
    batch_size=64,
    num_batches_per_epoch=100,
    target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    prediction_length=96,
    context_length=24*8,
    dequantize=False,
    freq='H',
    trainer=Trainer(
        device=device,
        epochs=200,
        learning_rate=5e-4,
        num_batches_per_epoch=100,
        batch_size=64,
    )
)
trained_net = estimator.train(dataset_train, num_workers=0, stage2_dict=stage1_downsampled_state_dict)
torch.save(trained_net.state_dict(), './Taxi_downsampled96_stage2.pkl')
