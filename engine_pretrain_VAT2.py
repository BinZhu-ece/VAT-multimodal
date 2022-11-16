# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import utils.misc as misc
import utils.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    """wogaide"""

    print('train_one_epoch!')
    for data_iter_step, samples in enumerate(data_loader):
        print(data_iter_step,'!!!!!!!!!!!!!!!')

        # # we use a per iteration (instead of per epoch) lr scheduler
        # if data_iter_step % accum_iter == 0:
        #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # # samples = samples.to(device, non_blocking=True)
        # data = samples
        # keys = data.keys()
        # data = {k: data[k].to(device) for k in keys}

        # with torch.cuda.amp.autocast():
        #     logits = model(**data)
        # loss = logits['loss']
        # print(f'loss type is {type(loss)}, loss is {loss}')
        # loss_value = loss.item()
        # print(f'loss_value type is {type(loss_value)}, loss is {loss_value}')

        # if not math.isfinite(loss_value):
        #     """wogaide"""
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)

        # loss /= accum_iter
        # """"wogaide"""

        # if not args.use_amp:# not args.use_amp
        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
        #     optimizer.step()
        # else:
        #     update_grad=(data_iter_step + 1) % accum_iter == 0  #"""https://zhuanlan.zhihu.com/p/500060805"""
        #     scaler = loss_scaler._scaler
        #     scaler.scale(loss).backward()
        #     if update_grad:
        #         scaler.unscale_(optimizer)
        #         norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
        #     scaler.step(optimizer)
        #     scaler.update()
        # # loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=args.clip_grad, model=model, update_grad=(data_iter_step + 1) % accum_iter == 0)
        # if (data_iter_step + 1) % accum_iter == 0:
        #     optimizer.zero_grad()

        # torch.cuda.synchronize()
        # # metric_logger.update(loss=loss_value)
        # lr = optimizer.param_groups[0]["lr"]
        # # metric_logger.update(lr=lr)
        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        # print(f'Rank:{args.rank},Epoch:{epoch},iter:{data_iter_step},lr:{lr},loss:{loss_value_reduce}')
        # # if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        # #     """ We use epoch_1000x as the x-axis in tensorboard.
        # #     This calibrates different curves when batch size changes.
        # #     """
        # #     epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        # #     log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
        # #     log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}