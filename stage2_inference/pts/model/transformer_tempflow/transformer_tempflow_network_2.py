import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gluonts.core.component import validated
from pts.modules import  MeanScaler, NOPScaler
from pts.modules.scaler import  StdScaler
from EncDecoder import Transformer_EncModel, Transformer_DecModel,  Encoder, Decoder, moving_avg, Past_Encoder, Past_Decoder
from codebook import Codebook
from gluonts.torch.modules.feature import FeatureEmbedder
from transformer import VQGANTransformer, Target_VQGANTransformer
class TransformerTempFlowTrainingNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        input_size: int,
        d_model: int,
        codebook_num: int,
        # input_scale,
        target_embed_dim: int,
        dropout_rate: float,
        n_heads: int,
        history_length: int,
        context_length: int,
        prediction_length: int,
        target_dim: int,
        factors: int,
        e_layers: int,
        d_layers: int,
        conditioning_length: int,
        latent_dim: int,
        codebook_beta: float,
        dequantize: bool,
        scaling: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_dim = target_dim
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.history_length = history_length
        self.scaling = scaling
        self.num_codebook = codebook_num
        self.beta_codebook = codebook_beta
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.d_model = d_model
        self.factors = factors
        self.embed_num = 64
        self.past_embed_num = 64
        self.target_embed_dim = target_embed_dim



        self.mov = moving_avg(kernel_size=4, stride=2)
        self.ratio = self.history_length // self.prediction_length


        self.input_projection = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size, out_channels=self.target_dim, kernel_size=3, stride=1, padding=1
            ),
        )

        for params in self.input_projection.parameters():
            params.requires_grad = False
        self.input_projection.eval()
        self.input_encoder = Encoder(dim_in=self.latent_dim)

        self.output_decoder = Decoder(dim_out=self.target_dim)

        self.d_ff = 4 * d_model
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_model = d_model

        self.time_proj =  nn.Linear(4, 4, bias=False)

        for params in self.time_proj.parameters():
            params.requires_grad = False
        self.time_proj.eval()

        self.encoder = Transformer_EncModel(d_model=self.latent_dim, n_heads=self.n_heads, d_ff=self.d_ff,
                                            factor=self.factors, pred_length=self.prediction_length,
                                            e_layers=self.e_layers, target_dim=self.target_dim)

        self.codebook = Codebook(num_codebook_vectors=self.num_codebook,
                                 latent_dim=self.latent_dim, beta=self.beta_codebook)

        self.quant_conv = nn.Conv1d(self.latent_dim, self.latent_dim, 1)
        self.post_quant_conv = nn.Conv1d(self.latent_dim, self.latent_dim, 1)

        self.embed_dim = 1
        self.embed = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        )

        for params in self.embed.parameters():
            params.requires_grad = False
        self.embed.eval()

        self.vqtransformer = VQGANTransformer(encoder=self.encoder, input_encoder=self.input_encoder,
                                              decoder=self.output_decoder,codebook=self.codebook, quant_conv=self.quant_conv,
                                              post_quant_conv=self.post_quant_conv, num_codebook=self.num_codebook,
                                              pkeep=0.8,target_dim=1, e_layers=4, d_layers=4, init_embed =self.latent_dim)

        self.stage2_pastencoder = Past_Encoder(dim_in=self.latent_dim)


        self.stage1_input_projection = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size, out_channels=self.target_dim, kernel_size=3, stride=1, padding=1
            ),
        )
        for params in self.stage1_input_projection.parameters():
            params.requires_grad = False
        self.stage1_input_projection.eval()

        self.stage1_input_encoder = Past_Encoder(dim_in=self.latent_dim)
        for params in self.stage1_input_encoder.parameters():
            params.requires_grad = False
        self.stage1_input_encoder.eval()

        self.stage1_time_proj = nn.Linear(4, 4, bias=False)
        for params in self.stage1_time_proj.parameters():
            params.requires_grad = False
        self.stage1_time_proj.eval()

        self.stage1_encoder = Transformer_EncModel(d_model=self.latent_dim, n_heads=self.n_heads, d_ff=self.d_ff,
                                                   factor=self.factors, pred_length=self.prediction_length,
                                                   e_layers=3, target_dim=self.target_dim)
        for params in self.stage1_encoder.parameters():
            params.requires_grad = False
        self.stage1_encoder.eval()


        self.stage1_codebook = Codebook(num_codebook_vectors=self.num_codebook,
                                        latent_dim=self.latent_dim, beta=self.beta_codebook)
        for params in self.stage1_codebook.parameters():
            params.requires_grad = False
        self.stage1_codebook.eval()

        self.stage1_quant_conv = nn.Conv1d(self.latent_dim, self.latent_dim, 1)

        for params in self.stage1_quant_conv.parameters():
            params.requires_grad = False
        self.stage1_quant_conv.eval()

        self.stage1_embed_dim = 1
        self.stage1_embed = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.stage1_embed_dim
        )

        for params in self.stage1_embed.parameters():
            params.requires_grad = False
        self.stage1_embed.eval()

        self.stage1_output_deocder = Past_Decoder(dim_out=self.target_dim)
        self.stage1_post_quant_conv = nn.Conv1d(self.latent_dim, self.latent_dim, 1)
        self.target_vqtransformer = Target_VQGANTransformer(encoder=self.stage1_encoder,
                                                     input_encoder=self.stage1_input_encoder,
                                                     decoder=self.stage1_output_deocder, codebook=self.stage1_codebook,
                                                     quant_conv=self.stage1_quant_conv,
                                                     post_quant_conv=self.stage1_post_quant_conv,
                                                     num_codebook=self.num_codebook,
                                                     pkeep=0.8, target_dim=1, e_layers=2, d_layers=3,
                                                     init_embed=self.latent_dim,
                                                     pred_length=48)

        self.reconstruction_loss = nn.MSELoss()
        self.dequantize = dequantize
        if self.scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)


    def create_network_input(
        self,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target_cdf: Optional[torch.Tensor],
        target_dimension_indicator: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Unrolls the RNN encoder over past and, if present, future data.
        Returns outputs and state of the encoder, plus the scale of
        past_target_cdf and a vector of static features that was constructed
        and fed as input to the encoder. All tensor arguments should have NTC
        layout.

        Parameters
        ----------
        past_time_feat
            Past time features (batch_size, history_length, num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        target_dimension_indicator
            Dimensionality of the time series (batch_size, target_dim)

        Returns
        -------
        outputs
            RNN outputs (batch_size, seq_len, num_cells)
        states
            RNN states. Nested list with (batch_size, num_cells) tensors with
        dimensions target_dim x num_layers x (batch_size, num_cells)
        scale
            Mean scales for the time series (batch_size, 1, target_dim)
        lags_scaled
            Scaled lags(batch_size, sub_seq_len, target_dim, num_lags)
        inputs
            inputs to the RNN

        """

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )
        sequence = future_target_cdf

        sequence_length = self.prediction_length

        index_embeddings = self.embed(target_dimension_indicator)

        time_embeddings = self.time_proj(future_time_feat)
        # sequence_length = self.history_length + self.prediction_length
        repeated_index_embeddings = (
            index_embeddings.unsqueeze(1)
            .expand(-1, sequence_length, -1, -1)
            .reshape((-1, sequence_length, index_embeddings.shape[1] * self.embed_dim))
        )

        _, scale = self.scaler(
            past_target_cdf,
            past_observed_values,
        )
        past_time_embedding = self.stage1_time_proj(past_time_feat)

        return sequence, time_embeddings, repeated_index_embeddings, past_target_cdf, past_time_embedding



    def distr_args(self, decoder_output: torch.Tensor):
        """
        Returns the distribution of DeepVAR with respect to the RNN outputs.

        Parameters
        ----------
        rnn_outputs
            Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)

        Returns
        -------
        distr
            Distribution instance
        distr_args
            Distribution arguments
        """
        (distr_args,) = self.proj_dist_args(decoder_output)

        # # compute likelihood of target given the predicted parameters
        # distr = self.distr_output.distribution(distr_args, scale=scale)

        # return distr, distr_args
        return distr_args

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target_cdf: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Computes the loss for training DeepVAR, all inputs tensors representing
        time series have NTC layout.

        Parameters
        ----------
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        future_observed_values
            Indicator whether or not the future values were observed
            (batch_size, prediction_length, target_dim)

        Returns
        -------
        distr
            Loss with shape (batch_size, 1)
        likelihoods
            Likelihoods for each time step
            (batch_size, context + prediction_length, 1)
        distr_args
            Distribution arguments (context + prediction_length,
            number_of_arguments)
        """



        inputs, time_embeds, index_embeds, past_target_cdf, past_time_feat = self.create_network_input(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            target_dimension_indicator=target_dimension_indicator)


        target_inputs = inputs


        mov_inputs = self.mov(inputs)
        mov_time_embeds = self.mov(time_embeds)
        mov_inputs = torch.cat((mov_inputs, mov_time_embeds), dim=-1)

        target_time_embeds = self.stage1_time_proj(future_time_feat)
        target_inputs = torch.cat((target_inputs, target_time_embeds), dim=-1)

        mov_inputs_proj = self.input_projection(mov_inputs.permute(0, 2, 1))
        target_inputs_proj = self.stage1_input_projection(target_inputs.permute(0, 2, 1))



        past_inputs = torch.cat((past_target_cdf, past_time_feat), dim=-1)
        # new added ##

        past_forward_inputs = past_inputs[:, :96, :]
        past_back_inputs = past_inputs[:, 96:, :]

        past_forward_projs = self.stage1_input_projection(past_forward_inputs.permute(0, 2, 1))
        past_forward_projs, _ = self.stage1_encoder(past_forward_projs.permute(0, 2, 1))

        past_back_projs = self.stage1_input_projection(past_back_inputs.permute(0, 2, 1))
        past_back_projs, _ = self.stage1_encoder(past_back_projs.permute(0, 2, 1))

        past_forward_outs = self.stage1_input_encoder(past_forward_projs)
        past_back_outs = self.stage1_input_encoder(past_back_projs)

        quant_past_forward_out = self.stage1_quant_conv(past_forward_outs)
        quant_past_back_out = self.stage1_quant_conv(past_back_outs)

        past_forward_codebook_mapping, past_forward_codebook_indices, _ = self.stage1_codebook(quant_past_forward_out)
        past_forward_indices = past_forward_codebook_indices.view(past_forward_codebook_mapping.shape[0], -1)

        past_back_codebook_mapping, past_back_codebook_indices, _ = self.stage1_codebook(quant_past_back_out)
        past_back_indices = past_back_codebook_indices.view(past_back_codebook_mapping.shape[0], -1)




        mov_indices = self.vqtransformer.sample_moving(x_past=past_back_indices, x_past_past=past_forward_indices)
        pred = self.target_vqtransformer.sample_final(past_back_indices, past_forward_indices, mov_indices)

        return pred, inputs




class TransformerTempFlowPredictionNetwork(TransformerTempFlowTrainingNetwork):
    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.pred_scaled = None


    def sampling_decoder(
        self,
        past_target_cdf: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        time_feat: torch.Tensor,
        past_time_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes sample paths by unrolling the RNN starting with a initial
        input and state.

        Parameters
        ----------
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        time_feat
            Dynamic features of future time series (batch_size, history_length,
            num_features)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)
        begin_states
            List of initial states for the RNN layers (batch_size, num_cells)
        Returns
        --------
        sample_paths : Tensor
            A tensor containing sampled paths. Shape: (1, num_sample_paths,
            prediction_length, target_dim).
        """

        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        repeated_past_inputs = repeat(past_target_cdf, dim=0)
        repeated_past_time = repeat(past_time_feat, dim=0)


        repeat_past_inputs = torch.cat((repeated_past_inputs, repeated_past_time), dim=-1)
        # new added ##

        repeat_past_forward_inputs = repeat_past_inputs[:, :96, :]
        repeat_past_back_inputs = repeat_past_inputs[:, 96:, :]

        repeat_past_forward_projs = self.stage1_input_projection(repeat_past_forward_inputs.permute(0, 2, 1))
        repeat_past_forward_projs, _ = self.stage1_encoder(repeat_past_forward_projs.permute(0, 2, 1))

        repeat_past_back_projs = self.stage1_input_projection(repeat_past_back_inputs.permute(0, 2, 1))
        repeat_past_back_projs, _ = self.stage1_encoder(repeat_past_back_projs.permute(0, 2, 1))

        repeat_past_forward_outs = self.stage1_input_encoder(repeat_past_forward_projs)
        repeat_past_back_outs = self.stage1_input_encoder(repeat_past_back_projs)

        repeat_quant_past_forward_out = self.stage1_quant_conv(repeat_past_forward_outs)
        repeat_quant_past_back_out = self.stage1_quant_conv(repeat_past_back_outs)

        repeat_past_forward_codebook_mapping, repeat_past_forward_codebook_indices, _ = self.stage1_codebook(repeat_quant_past_forward_out)
        repeat_past_forward_indices = repeat_past_forward_codebook_indices.view(repeat_past_forward_codebook_mapping.shape[0], -1)

        repeat_past_back_codebook_mapping, repeat_past_back_codebook_indices, _ = self.stage1_codebook(repeat_quant_past_back_out)
        repeat_past_back_indices = repeat_past_back_codebook_indices.view(repeat_past_back_codebook_mapping.shape[0], -1)

        mov_indices = self.vqtransformer.sample_moving(x_past=repeat_past_back_indices, x_past_past=repeat_past_forward_indices)

        samples = self.target_vqtransformer.sample_final(repeat_past_back_indices, repeat_past_forward_indices, mov_indices, sampling_shape=[5600, 256, 48])


        return samples.reshape(
            (
                -1,
                self.num_parallel_samples,
                self.prediction_length,
                self.target_dim,
            )
        )
    def create_predictor_input(
        self,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target_cdf: Optional[torch.Tensor],
        target_dimension_indicator: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unrolls the RNN encoder over past and, if present, future data.
        Returns outputs and state of the encoder, plus the scale of
        past_target_cdf and a vector of static features that was constructed
        and fed as input to the encoder. All tensor arguments should have NTC
        layout.

        Parameters
        ----------
        past_time_feat
            Past time features (batch_size, history_length, num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        target_dimension_indicator
            Dimensionality of the time series (batch_size, target_dim)

        Returns
        -------
        outputs
            RNN outputs (batch_size, seq_len, num_cells)
        states
            RNN states. Nested list with (batch_size, num_cells) tensors with
        dimensions target_dim x num_layers x (batch_size, num_cells)
        scale
            Mean scales for the time series (batch_size, 1, target_dim)
        lags_scaled
            Scaled lags(batch_size, sub_seq_len, target_dim, num_lags)
        inputs
            inputs to the RNN

        """

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )
        _, scale = self.scaler(
            past_target_cdf,
            past_observed_values,
        )

        past_time_embedding = self.stage1_time_proj(past_time_feat)

        return past_target_cdf, past_time_embedding

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target_cdf: torch.Tensor
    ) -> torch.Tensor:
        """
        Predicts samples given the trained DeepVAR model.
        All tensors should have NTC layout.
        Parameters
        ----------
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)

        Returns
        -------
        sample_paths : Tensor
            A tensor containing sampled paths (1, num_sample_paths,
            prediction_length, target_dim).

        """

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )
        past_target_cdf, past_time_embed = self.create_predictor_input(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=None,
            target_dimension_indicator=target_dimension_indicator,
        )


        return self.sampling_decoder(
            past_target_cdf=past_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
            time_feat=future_time_feat,
            past_time_feat=past_time_embed,
        )
