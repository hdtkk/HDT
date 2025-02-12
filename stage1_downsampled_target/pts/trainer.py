import time
from typing import List, Optional, Union

from tqdm.auto import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.optim import Adam
# from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from gluonts.core.component import validated


class Trainer:
    @validated()
    def     __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        beta1: float = 0.9,
        beta2: float = 0.95,
        disc_start: int=999999,
        maximum_learning_rate: float = 1e-2,
        clip_gradient: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.clip_gradient = clip_gradient
        self.device = device
        self.beta1 = beta1
        self.beta2 = beta2
        self.disc_start = disc_start
    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
    ) -> None:
        ############## Stage 2 ######################
        # stage2_params = []
        # for name, params in net.named_parameters():
        #     if params.requires_grad is True:
        #         # print('name', name)
        #         stage2_params.append(params)
        # optimizer = Adam(
        #    stage2_params, lr=self.learning_rate, eps=1e-08, betas=(self.beta1, self.beta2),
        #     weight_decay=self.weight_decay
        # )
        ################# Stage 2 ####################
        ########################## Stage 1 #############################
        optimizer = Adam(
            list(net.input_projection.parameters()) +
            list(net.input_encoder.parameters()) +
            list(net.output_decoder.parameters()) +
            list(net.time_proj.parameters()) +
            list(net.embed.parameters()) +
            list(net.encoder.parameters()) +
            # list(net.decoder.parameters()) +
            list(net.codebook.parameters()) +
            list(net.quant_conv.parameters()) +
            list(net.post_quant_conv.parameters()), lr=self.learning_rate,  eps=1e-08, betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay
        )
        #
        # # optimizer_dis = torch.optim.Adam(net.discriminator.parameters(),
        # #                             lr=5e-4, eps=1e-08, betas=(0.9, 0.95))
        #
        # # # lr_scheduler = OneCycleLR(
        # # #     optimizer,
        # # #     max_lr=self.maximum_learning_rate,
        # # #     steps_per_epoch=self.num_batches_per_epoch,
        # # #     epochs=self.epochs,
        # # # )
        ##############Stage 1################
        # optimizer_dis = torch.optim.Adam(net.discriminator.parameters(),
        #                                  lr=1e-4, eps=1e-08, betas=(self.beta1, self.beta2))
        iteration = 0
        batch_vq = 1
        batch_dis = 1
        ##############Stage 1################
        ############################# Stage 1##################################
        for epoch_no in range(self.epochs):
            # mark epoch start time
            tic = time.time()
            cumm_epoch_loss = 0.0
            cumm_epoch_dis_loss = 0.0
            total = self.num_batches_per_epoch - 1

            # training loop
            with tqdm(train_iter, total=total) as it:
                for batch_no, data_entry in enumerate(it, start=1):
                    optimizer.zero_grad()
                    # optimizer_dis.zero_grad()

                    inputs = [v.to(self.device) for v in data_entry.values()]
                    # print(inputs)
                    #
                    optim_index = (
                        0 if iteration < self.disc_start or iteration % 2 == 0 else 1
                    )

                    # lmbda = 0 if iteration < self.disc_start else None

                    output = net(*inputs)
                    if isinstance(output, (list, tuple)):
                        # X, Xhat,  D, Dhat, q_loss = output
                        X, Xhat, q_loss = output
                        # logits, target = output
                    else:
                        loss = output

                    # loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                    #                              target.reshape(-1))


                    ############ Stage 1 ############
                    criterion_vqvae = net.loss_fn
                    # criterion_discriminator = net.discriminator_loss
                    ##################### Stage 1##########

                    ###################### Stage 2 ###################
                    # cumm_epoch_loss += loss.item()
                    # avg_epoch_loss = cumm_epoch_loss / batch_no
                    # it.set_postfix(
                    #     {
                    #         "epoch": f"{epoch_no + 1}/{self.epochs}",
                    #         "avg_loss": avg_epoch_loss,
                    #         "Rec_loss": loss
                    #     },
                    #     refresh=False,
                    # )
                    #
                    # loss.backward()
                    #
                    # if self.clip_gradient is not None:
                    #     nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)
                    #
                    # optimizer.step()
                #     if batch_no
                # loss.backward
                    ######################## Stage 2################


                    ####################### Stage 1##################
                    if optim_index == 0:
                        # (4)
                       # BATCH_SIZE * 1 * DIM_T2
                       #  loss_rec, partial_losses = criterion_vqvae(
                       #      Xhat, X, Dhat, net.output_decoder, lmbda
                       #  )

                        loss_rec = criterion_vqvae(
                            Xhat, X
                        )
                        loss  = loss_rec + q_loss
                        loss.backward()
                        optimizer.step()
                    else:
                        # (3)
                        # D = self.discriminator(X.movedim(1, 2).detach())  # BATCH_SIZE * 1 * DIM_T2
                        # #  BATCH_SIZE * 1 * DIM_T2
                        # Dhat = self.discriminator(Xhat.movedim(1, 2).detach())
                        # Dhat_dis =  net.discriminator(Xhat.detach())
                        # D_dis = net.discriminator(X.detach())
                        Dhat_dis =  net.discriminator(Xhat.detach())
                        D_dis = net.discriminator(X.detach())
                        # loss, partial_losses = criterion_discriminator(Dhat, D)
                        loss, partial_losses = criterion_discriminator(Dhat_dis, D_dis)
                        loss.backward()
                        optimizer_dis.step()
                    # if self.clip_gradient is not None:
                    #     nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)
                    #
                    # optimizer.step()

                #     if self.num_batches_per_epoch == batch_no:
                #         break
                # it.close()

                    if optim_index == 0:
                        # loss_rec, loss_d, lamba = partial_losses
                        cumm_epoch_loss += loss.item()
                        avg_epoch_loss = cumm_epoch_loss / batch_vq
                        batch_vq = batch_vq + 1
                        iteration = iteration + 1
                        it.set_postfix(
                            {
                                "epoch": f"{epoch_no + 1}/{self.epochs}",
                                "avg_loss": avg_epoch_loss,
                                'iteration': iteration,
                                'rec_loss': loss_rec,
                                'q_loss': q_loss.item(),
                                # 'dis_loss': loss_d.item(),
                            },
                            refresh=False,
                        )
                    else:
                        cumm_epoch_dis_loss += loss.item()
                        avg_epoch_dis_loss = cumm_epoch_dis_loss / batch_dis
                        batch_dis = batch_dis + 1
                        iteration = iteration + 1
                        it.set_postfix(
                            {
                                "epoch": f"{epoch_no + 1}/{self.epochs}",
                                "avg_loss_1": avg_epoch_loss,
                                "avg_dis_loss": avg_epoch_dis_loss,
                                'rec_loss_1': loss_rec,
                                'q_loss_1': q_loss.item(),
                                'dis_loss_1': loss_d.item(),
                                'iteration': iteration,
                            },
                            refresh=False,
                        )
                ############### Stage 1 #################################
                #     # #
                #     # # cumm_epoch_loss += loss.item()
                #     # # avg_epoch_loss = cumm_epoch_loss / batch_no
                #     # # it.set_postfix(
                #     # #     {
                #     # #         "epoch": f"{epoch_no + 1}/{self.epochs}",
                #     # #         "avg_loss": avg_epoch_loss,
                #     # #     },
                #     # #     refresh=False,
                #     # # )
                #     #
                #     # loss.backward()
                #     # if self.clip_gradient is not None:
                #     #     nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)
                #     #
                #     # optimizer.step()
                #
                    if self.num_batches_per_epoch == batch_no:
                        break
                it.close()

            # validation loop
            if validation_iter is not None:
                cumm_epoch_loss_val = 0.0
                with tqdm(validation_iter, total=total, colour="green") as it:

                    for batch_no, data_entry in enumerate(it, start=1):
                        inputs = [v.to(self.device) for v in data_entry.values()]
                        with torch.no_grad():
                            output = net(*inputs)
                        if isinstance(output, (list, tuple)):
                            loss = output[0]
                        else:
                            loss = output

                        cumm_epoch_loss_val += loss.item()
                        avg_epoch_loss_val = cumm_epoch_loss_val / batch_no
                        it.set_postfix(
                            {
                                "epoch": f"{epoch_no + 1}/{self.epochs}",
                                "avg_loss": avg_epoch_loss,
                                "avg_val_loss": avg_epoch_loss_val,
                            },
                            refresh=False,
                        )

                        if self.num_batches_per_epoch == batch_no:
                            break

                it.close()

            # mark epoch end time and log time cost of current epoch
            toc = time.time()
