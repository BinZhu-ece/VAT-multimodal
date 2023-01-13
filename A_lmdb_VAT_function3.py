import argparse
import datetime
import json
import numpy as np
import os
import time

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
# import torchaudio
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
from os.path import join as opj
from transformers.trainer_utils import get_last_checkpoint
import os
from utils.datacollator import DataCollator
from models.crossformer import cross_former
from arguments.data import DataTrainingArguments
from arguments.model import ModelArguments
import transformers
from utils.fea_extractor import AudioFeatureExtractor
from utils.general import sample_frame_indices
# from engine_pretrain_VAT import train_one_epoch
import csv
import librosa
from utils.lmdb_class import lmdb_handle
import lmdb
from multiprocessing import Process

import time

def csv_read(csv_path='train.csv'):
        vat_list = []
        with open(csv_path, encoding="utf8") as f:
            csv_reader = csv.DictReader(f)
            for line in csv_reader:
                vat_list.append([line['video'],line['audio'],line['text']])
        return vat_list
    
def get_args_parser():
    parser = argparse.ArgumentParser('VAT-tokenization', add_help=False)
    # Dataset parameters
    parser.add_argument('--data_path', default='/remote-home/share/ImageNet-m/ImageNet2012/', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    #------------------------model start !--------------------------------------------
    parser.add_argument('--fusion_model_name_or_path', default='google/vit-base-patch16-224-in21k',
                        help='')
    parser.add_argument('--fusion_config_name', default='configs/crossformer/vit_config_tiny.json',
                        help='')
    parser.add_argument('--text_model_name_or_path', default='microsoft/deberta-v2-xlarge',
                        help='')
    parser.add_argument('--text_config_name', default='configs/text/bert_config_tiny.json',
                        help='')
    parser.add_argument('--tokenizer_name', default='microsoft/deberta-v2-xlarge',
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
    parser.add_argument('--audio_root', default='you_0_5',#you_0_5
                        help='')
    parser.add_argument('--text_root', default='you_0_5',#you_0_5
                        help='')
    parser.add_argument('--video_root', default='you_0_5',#you_0_5
                        help='')
    parser.add_argument('--train_file', default='test_11111.csv', help='')
    parser.add_argument('--validation_file', default='validation.csv',
                        help='')
    parser.add_argument('--per_device_train_batch_size', default=1,type=int,
                        help='')
    parser.add_argument('--contrastive_dim', default=768,type=int,
                        help='')
    parser.add_argument('--contrastive_loss_before_fusion', default=False, type=bool,
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
    #------------------------ lmdb start! ----------------------------------------------------
    parser.add_argument('--lmdb_va_ranks_folder', default=None,type=str,
                        help='')
    parser.add_argument('--part', default='0_5',type=str,
                        help='')
    #------------------------ lmdb over! ----------------------------------------------------
    
    return parser



def main(args):
    #------------------------------ DDP init --------------------------------------------
    if args.dist_url != "env://":
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

    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE']) 
        args.local_rank = int(os.environ['LOCAL_RANK'])

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method="env://",
                                         world_size=args.world_size, rank=args.rank)
        # torch.distributed.barrier()
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))

    # import torch.nn as nn
    # model = nn.Linear(1, 1, bias=False)
    # model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=False)#
    
    #------------------------------- training_logs start! -------------------------------------
    model_args=args
    data_args=args
    training_args=args
    audio_config = AutoConfig.from_pretrained(model_args.audio_config_name)
    audio_feature_extractor = AudioFeatureExtractor()
    
    text_config = AutoConfig.from_pretrained(model_args.text_config_name)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    ## -----------video-----------
    video_config = AutoConfig.from_pretrained(model_args.video_config_name)
    video_feature_extractor = VideoMAEFeatureExtractor.from_pretrained(model_args.videomae_feature_extractor_name, cache_dir=model_args.cache_dir)
    
    fusion_config = AutoConfig.from_pretrained(model_args.fusion_config_name)
    #------------------------------- training_logs over! -------------------------------------

    #---------------------------- video audio text function start!-------------------------------
    def audio_function(data_args,rela_path, waveform_len=audio_config.image_size*audio_config.image_size):
            def get_audio(path):
                waveform, sample_rate = librosa.load(path)
                # waveform = waveform.mean(0).numpy()
                length = len(waveform)
                if length >= waveform_len:
                    indices = sample_frame_indices(clip_len=waveform_len, frame_sample_rate=1, seg_len=length)
                    waveform = waveform[indices]
                else:
                    waveform = np.hstack([waveform for _ in range(waveform_len // length)] + [waveform[:waveform_len % length]])
                return waveform.reshape(1, audio_config.image_size, audio_config.image_size)

            waveform = get_audio(opj(data_args.audio_root,rela_path))

            pixel_values = audio_feature_extractor(waveforms=waveform, return_tensors="pt")['pixel_values']
            id = rela_path.split('.mp3')[0][-11:]
            return 'audio@{}'.format(id), pixel_values, id

    padding = "max_length" if data_args.pad_to_max_length else True
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 512:
            max_seq_length = 512
    else:
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)


    def tokenize_function(data_args, rela_path):
        text = open(opj(data_args.text_root, rela_path), 'r').read() 
        try:
            title = text.split('#')[0]
            if title[-1]==' ':
                title = title[:-1]
        except:
            print('This txt does not has a title!')
        
        # 时间长度，第一行（title+cls）, title_tokenization
        # return {'text_{}'.format(path): pixel_values}
        pixel_values = tokenizer(title, padding=padding, truncation=True, max_length=max_seq_length, return_attention_mask=True, return_tensors="pt")
        id = rela_path.split('.txt')[0][-11:]
        return 'text@{}'.format(id), pixel_values['input_ids'],id 
    def video_function(data_args,rela_path):
        def read_video(file_path):
            videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
            videoreader.seek(0)
            indices = sample_frame_indices(clip_len=video_config.num_frames, frame_sample_rate=4, seg_len=len(videoreader))
            video = videoreader.get_batch(list(indices)).asnumpy()
            return video

        video = list(read_video(opj(data_args.video_root,rela_path)))
        pixel_values = video_feature_extractor(video, return_tensors="pt")['pixel_values']

        id = rela_path.split('.mp4')[0][-11:] 
        return 'video@{}'.format(id), pixel_values, id
    #---------------------------- video audio text function over!-------------------------------
    
    

    vat_rela_path_list = csv_read(args.train_file)
    """because the lmdb.open(lock=false) for multiprocessing, so we must
        ensure the csv don't has the same video_id !!!!!!! 
    """
    # vat_rela_path_list = list(set(vat_rela_path_list))
    per_rank_num = len(vat_rela_path_list)//args.world_size + 1
    vat_rela_path_list.sort()# ensure the order of each rank

    

    def per_rank_proc_op(rank, per_rank_num, pro_, pros):

        assert args.part is not None, 'please add the youtube part number as args.part'
        assert args.lmdb_va_ranks_folder is not None, ' args.lmdb_va_ranks_folder is not None!'


        lmdb_video_path = opj(args.lmdb_va_ranks_folder,'youtube_video_part_{}_rank_{}_pro_{}'.format(args.part,args.rank,pro_))
        lmdb_audio_path = opj(args.lmdb_va_ranks_folder,'youtube_audio_part_{}_rank_{}_pro_{}'.format(args.part,args.rank,pro_))


        os.makedirs(args.lmdb_va_ranks_folder,exist_ok=True)
        # lmdb_vat=lmdb_handle(lmdb_vat_path)
        lmdb_video=lmdb_handle(lmdb_video_path)
        lmdb_audio=lmdb_handle(lmdb_audio_path)
        print('lmdb {} has been created!'.format(lmdb_video_path))
        print('lmdb {} has been created!'.format(lmdb_audio_path))


        cnt = 0
        vat_rela_path_list_rank = vat_rela_path_list[per_rank_num * (rank):per_rank_num * (rank + 1)]
        vat_rela_path_list_rank.sort()
        rank_len = len(vat_rela_path_list_rank)  # 每个GPU分的class数目
        per_pro = rank_len // pros + 1  # gpu上每个进程分的数目
        vat_rela_path_list_rank_pro = vat_rela_path_list_rank[per_pro * (pro_):per_pro * (pro_ + 1)]  # 指定GPU上每个pro分的class
        print(f'rank:{rank},pro_:{pro_},list_len:{len(vat_rela_path_list_rank_pro)}')
      
        
        cache_video = {}
        cache_audio = {}
        #------------------------------------------ video-audio-text write into lmdb start!-------------------------------------
        fail_cnt = 0
        success_cnt = 0

        t1 = time.time()
        for idx, vat_rela_path in enumerate(vat_rela_path_list_rank_pro):
            # [1.mp4,1.mp3,1.txt]
            # vat_rela_path = ['youtube_5/https___www_youtube_com_shorts_6YtjOlMfaqI.mp4', 'youtube_5/https___www_youtube_com_shorts_6YtjOlMfaqI.mp3', 'youtube_5/https___www_youtube_com_shorts_6YtjOlMfaqI.txt']           
            id_audio = 'audio@'+vat_rela_path[1].split('.mp3')[0][-11:]
            # id_video = 'video@'+vat_rela_path[0].split('.mp4')[0][-11:]

            if lmdb_audio.get(id_audio.encode()) is None:
                try:
                    #[video_rela_path, audio_rela_path , text_rela_path ]
                    # name_text, text_tokens, id_text  = tokenize_function(args, vat_rela_path[2])
                    name_video, video_tokens, id_video = video_function(args, vat_rela_path[0])
                    name_audio, audio_tokens, id_audio = audio_function(args, vat_rela_path[1])    
                    # assert id_text == id_audio == id_video, 'video-audio-text is not matched!!!!'
                    # cache[name_text]=text_tokens
                    cache_audio[name_audio]=audio_tokens
                    cache_video[name_video]=video_tokens
                    success_cnt += 1
                    if success_cnt%5==0:
                        print('rank {},pro {}, success:{}'.format(rank,pro_,success_cnt))
                    # print(name_video,'!!!!!')
                    assert len(cache_audio) == len(cache_video), 'video_cache nums is not equal to audio_cache!'
                    if len(cache_audio)%30==0:
                        
                        lmdb_video.add_tensors(cache_video)
                        lmdb_audio.add_tensors(cache_audio)
                        print('rank {},pro {}, the {}th batches, time cost:{}'.format(rank,pro_,idx//10, time.time()-t1))
                        t1 = time.time()

                        cache_video = {} 
                        cache_audio = {}   
                except Exception as e: 
                    print(vat_rela_path,e)
                    # MP4 error etc. 
                    print('Rank:{},pro_:{}, lmdb_vat_write fail in csv line {}'.format(rank,pro_,idx))
                    fail_cnt+=1
        
        assert len(cache_audio) == len(cache_video), 'video_cache nums is not equal to audio_cache! last!'     
        if len(cache_audio)!=0:
            lmdb_audio.add_tensors(cache_audio)
            lmdb_video.add_tensors(cache_video)
        print('Rank:{},pro_:{} has been finished!,lmdb_video stat is {}, lmdb_audio stat is {}'.format(args.rank,pro_, lmdb_video.stat(),lmdb_audio.stat()))
        lmdb_video.close()
        lmdb_audio.close()
        #------------------------------------------ video-audio-text write into lmdb over!-------------------------------------



    def per_rank_read(rank, per_rank_num, pros=None):
        # 上面是rank-gpu   下面是每个rank进行多进程
        process_list = []
        for pro_ in range(pros):  # 在最上面
            p = Process(target=per_rank_proc_op, args=([rank, per_rank_num, pro_, pros]))
            p.start()
            process_list.append(p)
        for j in process_list:
            p.join()

    per_rank_read(args.rank, per_rank_num, pros=48)

    # t_gather = torch.tensor([1+1j, 2+2j], dtype=torch.cfloat) + 2 * args.rank * (1+1j)
    # return t_gather
    # y = model(x)
    # dist.barrier()
    # per_rank_proc_op(0, per_rank_num, pro_=0, pros=1)


def test_lmdb(lmdb_vat_path):
    env=lmdb.open(lmdb_vat_path)
    txn = env.begin()
    print(txn.get('video@-oIXEtE11-0'.encode()))
    
    # {text:torch.LongTensor->np.int64, audio:torch.FloatTensor->np.float32, video: torch.FloatTensor->np.float32}
    for key,value in txn.cursor():
        key = key.decode()
        if key.startswith('text'):
            pixel_values = np.frombuffer(value,dtype=np.int64)#np.int32
        elif key.startswith('video') :
            pixel_values = np.frombuffer(value,dtype=np.float32)#np.int32
            # pixel_values = pixel_values.reshape(-1,16,3,224,224)
        elif key.startswith('audio'):
            pixel_values = np.frombuffer(value,dtype=np.float32)#np.int32
            # pixel_values = pixel_values.reshape(-1,1,224,224)

        pixel_values = torch.from_numpy(pixel_values)
        print(key,pixel_values)
        # break
    print(txn.stat())


if __name__ =='__main__':

    import os
    os.environ["TOKENIZERS_PARALLELISM"]  = "false"
    import warnings
    warnings.filterwarnings("ignore")
    # import pdb
    # pdb.set_trace()
    args = get_args_parser()
    args = args.parse_args()
    
    main(args)
    
    time.sleep(60*60*3)
    # t_gather = torch.tensor([1, 2], dtype=torch.cfloat).cuda(args.local_rank)
    # tensor_list = [torch.zeros(2, dtype=torch.cfloat).cuda(args.local_rank) for _ in range(args.world_size)] #.cuda(args.local_rank)
    # dist.all_gather(tensor_list, t_gather)
    # print(tensor_list)

    # print('11111111111')
    
    # dist.destroy_process_group()#销毁进程组
    # print('*'*50)
    # test_lmdb(args.lmdb_vat_folder)



    """
    python -m torch.distributed.launch --nproc_per_node 2  A_lmdb_VAT_function3.py  --train_file test_11111.csv \
    --video_root you_0_5 --audio_root you_0_5 --text_root you_0_5  --lmdb_va_ranks_folder  lmdb_va_0_5_ranks_folder
    """

