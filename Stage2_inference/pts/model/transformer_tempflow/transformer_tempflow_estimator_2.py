from typing import List, Optional,Iterable, Callable
from gluonts.itertools import Cyclic, PseudoShuffled, IterableSlice
from gluonts.transform import SelectFields
import torch
from gluonts.dataset.common import Dataset
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import TimeFeature
from utils2 import copy_parameters
from toolz import valmap
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.model.predictor import Predictor
from gluonts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    RemoveFields,
    RenameFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
    Identity,
)

from pts import Trainer
from pts.model import PyTorchEstimator
from pts.model.utils import get_module_forward_input_names
from pts.feature import (
    fourier_time_features_from_frequency
)

from .transformer_tempflow_network_2 import (
    TransformerTempFlowTrainingNetwork,
    TransformerTempFlowPredictionNetwork,
)


PREDICTION_INPUT_NAMES = ["past_target_cdf", "past_observed_values"]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target_cdf",
    "future_observed_values",
]

class TransformerTempFlowEstimator(PyTorchEstimator):
    @validated()
    def __init__(
        self,
        input_size: int,
        freq: str,
        prediction_length: int,
        e_layers: int,
        d_layers: int,
        target_dim: int,
        num_heads: int,
        target_embed_dim: int,
        # scaled,
        batch_size: int,
        num_batches_per_epoch: int,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        d_model: int = 32,
        codebook_num: int = 128,
        factors: int = 20,
        num_parallel_samples: int = 64,
        dropout_rate: float = 0.1,
        use_feat_dynamic_real: bool = False,
        latent_dim : float = 64,
        codebook_beta: float = 0.25,
        conditioning_length: int = 200,
        dequantize: bool = False,
        scaling: bool = True,
        pick_incomplete: bool = False,
        # lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        **kwargs,
    ) -> None:
        super().__init__(trainer=trainer, **kwargs)

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.input_size = input_size
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.codebook_beta = codebook_beta
        self.d_model = d_model
        self.codebook_num = codebook_num
        self.factors = factors
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        # self.act_type = act_type
        # self.dim_feedforward_scale = dim_feedforward_scale
        # self.num_encoder_layers = num_encoder_layers
        # self.num_decoder_layers = num_decoder_layers
        # self.state_list = state
        self.target_embed_dim = target_embed_dim
        self.num_parallel_samples = num_parallel_samples
        self.dropout_rate = dropout_rate
        
        self.use_feat_dynamic_real = use_feat_dynamic_real

        self.latent_dim = latent_dim
        # self.hidden_size = hidden_size
        self.e_layers = e_layers
        self.d_layers = d_layers
        # self.n_hidden = n_hidden
        self.n_heads = num_heads
        self.conditioning_length = conditioning_length
        self.dequantize = dequantize

        self.time_features = (
            time_features
            if time_features is not None
            else fourier_time_features_from_frequency(self.freq)
        )

        self.history_length = self.context_length
        self.pick_incomplete = pick_incomplete
        self.scaling = scaling

        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )

        self.validation_sampler = ValidationSplitSampler(
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )


    @classmethod
    def derive_auto_fields(cls, train_iter):
        stats = calculate_dataset_statistics(train_iter)

        return {
            "num_feat_dynamic_real": stats.num_feat_dynamic_real,
            "num_feat_static_cat": len(stats.feat_static_cat),
            "cardinality": [len(cats) for cats in stats.feat_static_cat],
        }

    def create_transformation(self) -> Transformation:
        remove_field_names = [FieldName.FEAT_DYNAMIC_CAT]
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [
                RemoveFields(field_names=remove_field_names),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=2,
                ),
                # maps the target to (1, T)
                # if the target data is uni dimensional
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=None,
                ),

                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME]
                                 + (
                                     [FieldName.FEAT_DYNAMIC_REAL]
                                     if self.use_feat_dynamic_real
                                     else []
                                 ),
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )

    def create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.history_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        ) + (
            RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                }
            )
        )

    def create_training_network(
        self, device: torch.device
    ) -> TransformerTempFlowTrainingNetwork:
        return TransformerTempFlowTrainingNetwork(
            input_size=self.input_size,
            target_dim=self.target_dim,
            codebook_num=self.codebook_num,
            d_model=self.d_model,
            n_heads = self.n_heads,
            factors = self.factors,
            e_layers = self.e_layers,
            d_layers = self.d_layers,
            target_embed_dim=self.target_embed_dim,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            scaling=self.scaling,
            latent_dim=self.latent_dim,
            codebook_beta=self.codebook_beta,
            conditioning_length=self.conditioning_length,
            dequantize=self.dequantize,
        ).to(device)


    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: TransformerTempFlowTrainingNetwork,
        device: torch.device,
    ) -> Predictor:
        prediction_network = TransformerTempFlowPredictionNetwork(
            input_size=self.input_size,
            target_dim=self.target_dim,
            codebook_num=self.codebook_num,
            target_embed_dim=self.target_embed_dim,
            d_model=self.d_model,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            scaling=self.scaling,
            n_heads=self.n_heads,
            factors=self.factors,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            latent_dim=self.latent_dim,
            codebook_beta=self.codebook_beta,
            conditioning_length=self.conditioning_length,
            dequantize=self.dequantize,
            num_parallel_samples=self.num_parallel_samples,
        ).to(device)
        copy_parameters(trained_network, prediction_network)
        input_names = get_module_forward_input_names(prediction_network)
        prediction_splitter = self.create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            # freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )
