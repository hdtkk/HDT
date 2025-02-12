import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from gluonts.core.component import validated
from pts.modules import  MeanScaler, NOPScaler
from pts.modules.scaler import  StdScaler
from EncDecoder import Transformer_EncModel, Transformer_DecModel,\
    New_FullAttention, Discriminator,  Encoder, Decoder, VQGANloss, DiscriminatorLoss, moving_avg
from codebook import Codebook
from gluonts.torch.modules.feature import FeatureEmbedder
from transformer import VQGANTransformer
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

        # target_embed
        self.factors = factors
        self.embed_num = 64
        self.past_embed_num = 64
        self.target_embed_dim = target_embed_dim



        self.mov = moving_avg(kernel_size=4, stride=2)
        self.ratio = self.history_length // self.prediction_length
        #

        self.input_projection = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size, out_channels=self.target_dim, kernel_size=3, stride=1, padding=1
            ),
        )

        # self.past_input_projection = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=input_size, out_channels=self.target_dim, kernel_size=3, stride=1, padding=1
        #     ),
        # )

        # self.input_projection = nn.Linear(input_size, self.d_model, bias=True)
        # for param in self.input_encoder.parameters():
        #     param.requires_grad = False
        #
        # self.input_encoder.eval()
        # self.past_input_encoder = Encoder(dim_in=self.latent_dim)

        self.input_encoder = Encoder(dim_in=self.latent_dim)



        self.output_decoder = Decoder(dim_out=self.target_dim)

        self.d_ff = 4 * d_model
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_model = d_model

        self.time_proj =  nn.Linear(4, 4, bias=False)

        self.encoder = Transformer_EncModel(d_model=self.latent_dim, n_heads=self.n_heads, d_ff=self.d_ff,
                                            factor=self.factors, pred_length=self.prediction_length,
                                            e_layers=self.e_layers, target_dim=self.target_dim)
        #


        self.codebook = Codebook(num_codebook_vectors=self.num_codebook,
                                 latent_dim=self.latent_dim, beta=self.beta_codebook)

        self.quant_conv = nn.Conv1d(self.latent_dim, self.latent_dim, 1)
        #self.quant_conv.eval()
        self.post_quant_conv = nn.Conv1d(self.latent_dim, self.latent_dim, 1)

        self.embed_dim = 1
        self.embed = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        )

        # self.discriminator = Discriminator(target_dim=self.target_dim, d_model=self.d_model)

        # self.vqtransformer = VQGANTransformer(embed_model=self.embed_layer, encoder=self.encoder, input_encoder=self.input_encoder,
        #                                       decoder=self.decoder,codebook=self.codebook, quant_conv=self.quant_conv,
        #                                       post_quant_conv=self.post_quant_conv, num_codebook=self.num_codebook,
        #                                       pkeep=1.0,target_dim=1, e_layers=3, d_layers=3, init_embed =self.d_model)
        # self.vqtransformer.eval()
        ################ Stage 1 ############
        # self.discriminator_loss = DiscriminatorLoss(weight=0.8)
        # self.discriminator = Discriminator(target_dim=self.target_dim, d_model=self.d_model)
        self.loss_fn = VQGANloss(alpha=1.0, gamma=1e-4, discriminator_weight=0.8)
        # ############### Stage 1 #########
        self.dequantize = dequantize
        if self.scaling:
            self.scaler = MeanScaler(keepdim=True)
            # self.scaler = StdScaler(keepdim=True)
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

        # print('past_observed_values', past_observed_values.shape)
        # print('past_is_pad', past_is_pad.shape)
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        sequence = future_target_cdf
        # print('sequence', sequence.shape)
        # dasd
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
        # past_target_cdf = past_target_cdf / scale

        return sequence, time_embeddings, repeated_index_embeddings, past_target_cdf, past_time_feat



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


        # inputs = self.mov(inputs)
        # time_embeds = self.mov(time_embeds)
        # print(inputs.shape)
        # das

        target_inputs = inputs

        inputs = torch.cat((inputs, time_embeds), dim=-1)


        # inputs = torch.cat((test_inputs, test_time), dim=-1)

        inputs_proj = self.input_projection(inputs.permute(0, 2, 1))

        batch, latent_dim, pred_length = inputs_proj.shape

        #####past_encode#########

        # past_inputs = torch.cat((past_target_cdf, past_time_feat), dim=-1)
        # past_proj = self.past_input_projection(past_inputs.permute(0, 2, 1))
        # past_outs = self.past_input_encoder(past_proj)

        ######past_encode ###########
        inputs_proj, enc_attns = self.encoder(inputs_proj.permute(0, 2, 1))

        # enc_inputs = self.input_encoder(inputs_proj)
        enc_outs = self.input_encoder(inputs_proj)
        # print('enc_input', enc_inputs.shape)


        _, _, enc_length = enc_outs.shape
        ########################### Stage 1 combined ##########################
        # batch, p_l, latent_d = x_inputs.shape
        #
        # enc_outs, enc_attns = self.encoder(enc_outs.permute(0, 2, 1))

        quant_enc_out = self.quant_conv(enc_outs)  # b, d_model, target_dim * pred_length

        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_enc_out)

        post_quant_conv_out = self.post_quant_conv(codebook_mapping)

        # post_quant_conv_out, dec_attns= self.decoder(
        #     post_quant_conv_out.permute(0, 2, 1))  # B, L, ts_dim
        # print(post_quant_conv_out.shape)
        dec_output = self.output_decoder(post_quant_conv_out).permute(0, 2, 1)
        # dec_output = self.output_decoder(dec_output.permute(0, 2, 1)).permute(0, 2, 1)
        # Dhat = self.discriminator(dec_output)
        # D = self.discriminator(enc_inputs[:, :, :self.target_dim])
        # D = self.discriminator(target_inputs)

        return (target_inputs, dec_output, q_loss)
        # return (target_inputs, dec_output, D, Dhat, q_loss)
        # return (discriminator_inputs, dec_output, q_loss)
        ########################## Stage 1 combined ####################


        # for i in range(self.embed_num):
        #     solar_embed.append(self.embed_layer[i](solar_inputs))
        # x_solar = torch.cat(solar_embed, dim=1)
        # # print('x_solar', x_solar[0])
        # for i in range(self.embed_num):
        #     ele_embed.append(self.embed_layer[i](ele_inputs))
        # x_ele = torch.cat(ele_embed, dim=1)

        # print('x_ele', x_ele[0])
        # dasd
        # batch, p_l, latent_d = x_inputs.shape

        # x_enc_solar = self.input_encoder(x_solar).permute(0, 2, 1)
        # x_enc_ele = self.input_encoder(x_ele).permute(0, 2, 1)



        # x_inputs_solar, _ = self.embedding_cross(x_solar) # b, l, d_model
        # x_inputs_ele, _ = self.embedding_cross(x_ele)
        # x_enc_solar = self.input_encoder(x_inputs_solar.permute(0, 2, 1)).permute(0, 2, 1)
        # # print(x_enc_solar.shape)
        # x_enc_ele = self.input_encoder(x_inputs_ele.permute(0, 2, 1)).permute(0, 2, 1)
        # print(x_enc_ele.shape)
        # x_inputs = torch.cat((x_inputs_solar, x_inputs_ele), dim=-1) # b, l , 2 * d_model
        # x_inputs = self.input_encoder(x_inputs)
        # dasd
        # batch, p_l, latent_d = x_enc_ele.shape # b, L , embed_num


        # solar_logits, solar_target = self.vqtransformer(x_enc_solar, solar_past)

        # ele_logits, ele_target = self.vqtransformer(x_enc_ele, ele_past)
        # return ele_logits, ele_target
        # return solar_logits, solar_target, ele_logits, ele_target

        # batch, p_l, latent_d = x_inputs.shape
        ############################################# Stage 1 ###################################
        # # # print(dec_output.shape)
        # # Dhat = self.discriminator(dec_output)
        # # D = self.discriminator(dis_inputs)
        # # # dsad
        # # print(dec_output_solar.shape)
        # for i in range(64):
        #     # test_solar = solar_inputs[i].permute(1, 0).squeeze(-1)
        #     # target_solar = dec_output_solar[i].squeeze(-1)
        #     # print('test_solar', test_solar)
        #     # print('target_solar', target_solar)
        #     # test_ele = ele_inputs[i].permute(1, 0).squeeze(-1)
        #     # target_ele = dec_output_ele[i].squeeze(-1)
        #     # print('test_ele', test_ele)
        #     # print('target_ele', target_ele)
        #     loss_solar = torch.sum((dec_output_solar[i] - solar_inputs[i].permute(1, 0)), dim=(0,1)) / (
        #             24)
        #     loss_ele =  torch.sum((dec_output_ele[i] - ele_inputs[i].permute(1, 0)), dim=(0,1)) / (
        #             24)
        #     # print('pred', dec_output_ele[i].transpose(0, 1))
        #     # print('taget', ele_inputs[i])
        #     print('solar', loss_solar)
        #     print('ele', loss_ele)
        # dasdad

        # # embedded_inputs = [self.embedding_layers(expanded_inputs[:, :, i, :]) for i, embedding_layer in
        # #                    enumerate(self.embedding_layers)]
        # # embedded_input = torch.stack(embedded_inputs, dim=2)  # # B, pred_length, target_dim, embed_dim
        # #################################################################
        # if self.dequantize:
        #     future_target_cdf += torch.rand_like(future_target_cdf)
        # # return (inputs, dec_output, q_loss)
        # q_loss = q_loss_solar + q_loss_ele
        #############################################Stage 1 #########################################
        # return (dis_inputs, dec_output, D, Dhat, q_loss)




class TransformerTempFlowPredictionNetwork(TransformerTempFlowTrainingNetwork):
    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.pred_scaled = None
        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to
        # the last target value
        # self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        past_target_cdf: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        time_feat: torch.Tensor,
        scale: torch.Tensor,
        past_inputs: torch.Tensor,
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

        # blows-up the dimension of each tensor to
        # batch_size * self.num_sample_paths for increasing parallelism
        # repeated_past_target_cdf = repeat(past_target_cdf)
        # repeated_time_feat = repeat(time_feat)

        # repeated_scale = repeat(scale)

        # if self.scaling:
        #     self.flow.scale = repeated_scale
        # repeated_target_dimension_indicator = repeat(target_dimension_indicator)
        # scaled 64, 1, 370
        # 7* 64, 1, 370
        # repeated_scale = scale.repeat_interleave(repeats=7, dim=0)


        repeated_past_inputs = repeat(past_inputs, dim=0)
        # print(repeated_past_inputs[:20].shape)
        # dasd
        # future_samples = []
        # print(self.num_parallel_samples)
        sampling_shape = [7 * self.num_parallel_samples, self.latent_dim, 12]
        samples = self.vqtransformer.log_series(x_past=repeated_past_inputs, sampling_shape=sampling_shape)
        # print('samples', samples.shape)
        # print('samples', samples[:64])
        sample_copy = torch.zeros_like(samples).to(samples.device)

        return sample_copy.reshape(
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,]:
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


        time_feat = torch.cat(
            (
                past_time_feat[:, -self.context_length:, ...],
                future_time_feat,
            ),
            dim=1,
        )
        _, scale = self.scaler(
            past_target_cdf,
            past_observed_values,
        )

        past_scaled = past_target_cdf / scale

        index_embeddings = self.embed(target_dimension_indicator)
        # assert_shape(index_embeddings, (-1, self.target_dim, self.embed_dim))

        # (batch_size, seq_len, target_dim * embed_dim)
        repeated_index_embeddings = (
            index_embeddings.unsqueeze(1)
            .expand(-1, self.context_length, -1, -1)
            .reshape((-1, self.context_length, self.target_dim * self.embed_dim))
        )
        inputs = torch.cat((past_scaled, repeated_index_embeddings, past_time_feat), dim=-1)


        return inputs, self.pred_scaled, index_embeddings

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

        # mark padded data as unobserved
        # (batch_size, target_dim, seq_len)
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )
        # print('3', future_target_cdf)
        inputs, scale, static_feat = self.create_predictor_input(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=None,
            target_dimension_indicator=target_dimension_indicator,
        )
        # dsad
        # inputs 7, 96, 744    scale
        # print('input', inputs.shape)
        # print('scaled', scale.shape)


        # pred_past_in = inputs[:, :self.context_length, ...]
        pred_past_inputs = self.past_input(inputs)
        #
        # enc_out = self.transformer.encoder(self.encoder_input(inputs).permute(1, 0, 2))

        return self.sampling_decoder(
            past_target_cdf=past_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
            time_feat=future_time_feat,
            scale=scale,
            past_inputs=pred_past_inputs,
        )
