# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
import timm
import random

# wogaide
# assert timm.__version__ == "0.3.2"  # version check

import timm.optim.optim_factory as optim_factory
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

import sys
import numpy as np
import torch
from decord import VideoReader, cpu
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, VideoMAEFeatureExtractor, Trainer, HfArgumentParser, \
    set_seed, is_torch_tpu_available, AutoConfig, AutoTokenizer, VideoMAEConfig, \
    VideoMAEForPreTraining, ViTMAEForPreTraining, ViTMAEConfig, ViTModel
from datasets import load_dataset
import torchvision.transforms as V_T
# import torchaudio.transforms as A_T
import librosa
from os.path import join as opj
from transformers.trainer_utils import get_last_checkpoint
import os
import logging
from utils.datacollator import DataCollator
from models.crossformer import cross_former
from arguments.data import DataTrainingArguments
from arguments.model import ModelArguments
from arguments.train import TrainArguments
import datasets
import transformers
from utils.fea_extractor import AudioFeatureExtractor
from utils.general import sample_frame_indices
from engine_pretrain_VAT import train_one_epoch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# time.sleep(10000)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # parser.add_argument('--batch_size', default=24, type=int,
    #                     help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_pixel2', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.9, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',default=True,#--norm_pix_loss
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/remote-home/share/ImageNet-m/ImageNet2012/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')

    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
                        
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    # parser.set_defaults(pin_mem=True)

    parser.add_argument('--use_amp', default=True, 
                        help='')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--clip_grad', default=0.5, type=float)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')#--dist_on_itp   ddp
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True,
                        help='url used to set up distributed training')
    parser.add_argument('--gpus', default=[0,1,2,3],
                        help='DP CUDA devices')


    #------------------------model start !--------------------------------------------
    parser.add_argument('--fusion_model_name_or_path', default='google/vit-base-patch16-224-in21k',
                        help='')
    parser.add_argument('--fusion_config_name', default='configs/crossformer/vit_config_tiny.json',
                        help='')
    parser.add_argument('--text_model_name_or_path', default='bert-base-uncased',
                        help='')
    parser.add_argument('--text_config_name', default='configs/text/bert_config_tiny.json',
                        help='')
    parser.add_argument('--tokenizer_name', default='bert-base-uncased',
                        help='')
    parser.add_argument('--audio_model_name_or_path', default='facebook/vit-mae-base',
                        help='')
    parser.add_argument('--audio_config_name', default='configs/audio/mae_config_tiny.json',
                        help='')
    parser.add_argument('--mae_feature_extractor_name', default='facebook/vit-mae-base',
                        help='')
    parser.add_argument('--video_model_name_or_path', default='MCG-NJU/videomae-base',
                        help='')
    parser.add_argument('--video_config_name', default='configs/video/videomae_config_tiny.json',
                        help='')
    parser.add_argument('--videomae_feature_extractor_name', default='MCG-NJU/videomae-base',
                        help='')
    parser.add_argument('--cache_dir', default='cache_dir',
                        help='')
    #------------------------model over-----------------------------------------------------


    #------------------------ data start! --------------------------------------------------
    parser.add_argument('--audio_root', default='datasets/audio',
                        help='')
    parser.add_argument('--text_root', default='datasets/text',
                        help='')
    parser.add_argument('--video_root', default='datasets/video',
                        help='')
    parser.add_argument('--train_file', default='train.csv',
                        help='')
    parser.add_argument('--validation_file', default='validation.csv',
                        help='')
    parser.add_argument('--per_device_train_batch_size', default=1,type=int,
                        help='')
    parser.add_argument('--contrastive_dim', default=768,type=int,
                        help='')
    parser.add_argument('--contrastive_loss_before_fusion', default=False,type=bool,
                        help='')
    parser.add_argument('--max_seq_length', default=None,
                        help='')
    parser.add_argument('--pad_to_max_length', default=False,
                        help='')
    parser.add_argument('--max_train_samples', default=None,
                        help='')
    parser.add_argument('--do_train', default=True,
                        help='')
    parser.add_argument('--mlm_probability', default=0.3,type=float,
                        help='')
    parser.add_argument('--video_mask_ratio', default=0.8,type=float,
                        help='')
    parser.add_argument('--audio_mask_ratio', default=0.3,type=float,
                        help='')
    parser.add_argument('--vt_match_ratio', default=0.5,type=float,
                        help='')
    #------------------------ data over! ----------------------------------------------------
    return parser

def main(args):
    #------------------------------- training_logs start! -------------------------------------
    # logger = logging.getLogger(__name__)
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    # model_args, data_args, _ = parser.parse_args_into_dataclasses()
    # import logging
    # logger = logging.getLogger(__name__)
    # logger.setLevel("warning")
    model_args=args
    data_args=args
    training_args=args
    #------------------------------- training_logs over! -------------------------------------
    #-------------------------------------进程组 start!---------------------------------------
    dist.init_process_group(
    backend='nccl',
    init_method=args.dist_url,
    world_size=args.world_size,
    rank=args.rank
    )#初始化
    assert dist.is_initialized()
    if args.rank==0:
        print('进程组初始化完成')
    set_seed(args.world_size+args.seed)
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    # cudnn.benchmark = True
    #-------------------------------------进程组 over!---------------------------------------
    #--------------------------------- prepare model start!---------------------------------
    ## -----------audio-----------
    audio_config = AutoConfig.from_pretrained(model_args.audio_config_name)
    audio_config.mask_ratio = data_args.audio_mask_ratio
    audio_feature_extractor = AudioFeatureExtractor()
    ## -----------text-----------
    text_config = AutoConfig.from_pretrained(model_args.text_config_name)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    ## -----------video-----------
    video_config = AutoConfig.from_pretrained(model_args.video_config_name)
    video_feature_extractor = VideoMAEFeatureExtractor.from_pretrained(model_args.videomae_feature_extractor_name, cache_dir=model_args.cache_dir)
    ## -----------fusion-----------
    fusion_config = AutoConfig.from_pretrained(model_args.fusion_config_name)
    model = cross_former(model_args, audio_config, text_config, video_config, fusion_config)
    model.cuda(training_args.local_rank)
    model= torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print('BN同步完成')
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[training_args.local_rank],output_device=training_args.local_rank,find_unused_parameters=False)#
    model_without_ddp = model.module
    #--------------------------------- prepare model over!---------------------------------
    #--------------------------------- read lmdb start -----------------------------------
    def read_lmdb(id_):
        # {text:torch.LongTensor->np.int64, audio:torch.FloatTensor->np.float32, video: torch.FloatTensor->np.float32}
        key = id_.encode()
        if id_.startswith('video'):
            # id_video = 'video@'+rela_path.split('.mp4')[0][-11:]
            
            value = txn_begin.get(key)
            pixel_values = np.frombuffer(value,dtype=np.float32)#np.int32
            pixel_values = pixel_values.reshape(-1,16,3,224,224)
            
            pixel_values = torch.from_numpy(pixel_values)
            # print('video :',pixel_values.shape)
            return pixel_values

        elif id_.startswith('.mp3') or id_.startswith('.wav'):
            # if rela_path.endswith('.mp3'):
            #     id_audio = 'audio@'+rela_path.split('.mp3')[0][-11:]
            # else:
            #     id_audio = 'audio@'+rela_path.split('.wav')[0][-11:]
            
            value = txn_begin.get(key)
            pixel_values = np.frombuffer(value,dtype=np.float32)#np.int32
            pixel_values = pixel_values.reshape(-1,1,224,224)
            
            pixel_values = torch.from_numpy(pixel_values)
            # print('audio :',pixel_values.shape)
            return pixel_values

        elif id_.startswith('.txt'):
            # id_text = 'text@'+rela_path.split('.txt')[0][-11:]
            
            value = txn_begin.get(key)
            pixel_values = np.frombuffer(value,dtype=np.int64)#np.int32
            pixel_values = torch.from_numpy(pixel_values)
            return pixel_values

    #--------------------------------- read lmdb over --------------------------------------
    #--------------------------------- prepare tokenizer start!---------------------------------
    def audio_function(examples, waveform_len=audio_config.image_size*audio_config.image_size):

        pixel_values = torch.concat(  [ read_lmdb(i) for i in examples['audio'] ]   ,0)
        # print('concat audio:',pixel_values.shape)
        #torch.Size([n, 1, 224, 224])
        return {'audio_pixel_values': pixel_values}

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 512:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 512 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 512
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    padding = "max_length" if data_args.pad_to_max_length else True

    def tokenize_function(examples):
        def read2(data_args, rela_path):
            text = open(opj(data_args.text_root, rela_path), 'r').read() 
            title = text.split('#')[0]
            if title[-1]==' ':
                title = title[:-1]
            return title
        # torch.Size([4, 512])
        text = [read2(data_args,t) for t in examples['text']]
        return tokenizer(text, padding=padding, truncation=True, max_length=max_seq_length, return_attention_mask=True, return_tensors="pt")
        # text = [open(opj(data_args.text_root, t), 'r').read() for t in examples['text']]
        # return tokenizer(text, padding=padding, truncation=True, max_length=max_seq_length, return_attention_mask=True, return_tensors="pt")

    def video_function(examples):
        # torch.Size([4, 16, 3, 224, 224])
        # print(examples['video'],111)
        pixel_values = torch.concat(  [  read_lmdb(v) for v in examples['video']]  , 0 )
        # video = [list(read_video(opj(data_args.video_root, v))) for v in examples['video']]
        # pixel_values = video_feature_extractor(video, return_tensors="pt")['pixel_values']

        return {'video_pixel_values': pixel_values }

    #--------------------------------- prepare tokenizer over!---------------------------------


    #----------------------------- prepare dataset start!----------------------------------
    dataset = load_dataset('csv', data_files={'train': [data_args.train_file], },
                           cache_dir=model_args.cache_dir)
    # 'validation': [data_args.validation_file]
    if training_args.rank==0:
        print(f'train_dataset is {data_args.train_file}')

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.map(audio_function, batched=True)
    dataset = dataset.map(video_function, batched=True)

    synchronize()#同步

    assert training_args.do_train is True, 'training_args.do_train is False'
    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        train_dataset = train_dataset.remove_columns(['audio', 'text', 'video', 'token_type_ids'])
    synchronize()#同步
    print('train_dataset is succeed!')
    # if training_args.do_eval:
    #     if "validation" not in dataset:
    #         raise ValueError("--do_eval requires a validation dataset")
    #     eval_dataset = dataset["validation"]
    #     eval_dataset = eval_dataset.remove_columns(['audio', 'text', 'video', 'token_type_ids'])
    data_collator = DataCollator(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability, vt_match_ratio=data_args.vt_match_ratio,
                                 num_frames=video_config.num_frames, video_image_size=video_config.image_size, video_patch_size=video_config.patch_size,
                                 tubelet_size=video_config.tubelet_size, video_mask_ratio=data_args.video_mask_ratio)
    print('data_collator is succeed!')
    """DDP dataloader"""
    if args.distributed:  # args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=training_args.world_size,
            rank=training_args.rank,
            shuffle=True,
            )
    print('data_sampler is succeed!')
    synchronize()#同步
    if training_args.rank==0:
        print("Sampler_train = %s" % str(train_sampler))
    train_loader = DataLoader(train_dataset, 
                            batch_size=training_args.per_device_train_batch_size,  
                            collate_fn=data_collator, 
                            sampler=train_sampler, 
                            num_workers=training_args.num_workers, 
                            pin_memory=args.pin_mem, drop_last=True )
 
    #-------------------------------- prepare dataset over ! -------------------------------------
    
    #-------------------------------- default setting start! ---------------------------------
    eff_batch_size = args.per_device_train_batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    if args.rank==0:
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)
        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)
    synchronize()#同步
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=training_args.lr, betas=(0.9, 0.95))
    if args.rank==0:
        print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if training_args.rank==0:
        print(f"Start training for {training_args.epochs} epochs")
    start_time = time.time()
    if args.rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
    for epoch in range(training_args.start_epoch, training_args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,train_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=training_args,     
        )
        if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs) and args.rank==0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
                        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    if args.rank==0:
        print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    args = get_args_parser()
    args = args.parse_args()
    main(args)
