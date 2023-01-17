"""
    python3 Step0_construct_vat_filename_csv.py --video_parts_folder ./test_coco  (./不要少，或者绝对路径)
    
    将文件夹下面的文件提前保存为csv,方便后边构建video-audio-text feature lmdb.
    
    video_parts_folder(之前用的文件夹里面是小文件夹，后来文件夹里面直接是文件了):
       1.mp4
       1.m4a
       1.json
       2.mp4
       .....
    
"""


import os
import glob
import os.path as osp
import csv
import argparse
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser('vat_rela_path csv construct', add_help=False)
    parser.add_argument('--video_parts_folder', default='./test_coco', type=str)
    parser.add_argument('--part', default=1000, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    #------------------------ data over! ----------------------------------------------------
    return parser



def simple_construct(args):
    audio_raw_path_list_all = sum([glob.glob(
            osp.join(args.video_parts_folder,'*.m4a'),
            recursive=True)], [])
    audio_raw_path_list_all.sort()

    # 1. 创建文件对象
    f = open('{}_filename.csv'.format(args.video_parts_folder),'w',encoding='utf-8')

    # 2. 基于文件对象构建 csv写入对象
    csv_writer = csv.writer(f)

    # 3. 构建列表头
    csv_writer.writerow(["audio","text","video"])

    # 4. 写入csv文件内容
    for audio_path in audio_raw_path_list_all:
        audio_rela_path = audio_path.split('/')[-1]
        video_rela_path = audio_rela_path.replace('m4a','mp4')
        text_rela_path = audio_rela_path.replace('m4a','json')

        csv_writer.writerow([audio_rela_path,text_rela_path,video_rela_path])
    # 5. 关闭文件
    f.close()
    
def main(args):
    simple_construct(args)
   

if __name__=="__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
    
    
    
    def complex_construct(args):
        video_depth_path_list_all = sum([glob.glob(
            osp.join(args.video_parts_folder, 'keywords_youtube_url_json_folder_part{}_60_download_video'.format(args.part),
                    'you*','*depth.mp4'),
            recursive=True)], [])
        video_thermal_path_list_all = sum([glob.glob(
            osp.join(args.video_parts_folder, 'keywords_youtube_url_json_folder_part{}_60_download_video'.format(args.part),
                    'you*','*thermal.mp4'),
            recursive=True)], [])
        video_all_path_list_all = sum([glob.glob(
            osp.join(args.video_parts_folder, 'keywords_youtube_url_json_folder_part{}_60_download_video'.format(args.part),
                    'you*','*.mp4'),
            recursive=True)], [])

        video_raw_path_list_all = list(set(video_all_path_list_all)-set(depth_video_path_list_all)-set(video_thermal_path_list_all)-set(video_depth_path_list_all))  #没有depth的video
        audio_raw_path_list_all = sum([glob.glob(
            osp.join(args.video_parts_folder, 'keywords_youtube_url_json_folder_part{}_60_download_video'.format(args.part),
                    'you*','*.mp3'),
            recursive=True)], [])
        txt_raw_path_list_all = sum([glob.glob(
            osp.join(args.video_parts_folder, 'keywords_youtube_url_json_folder_part{}_60_download_video'.format(args.part),
                    'you*','*.txt'),
            recursive=True)], [])

        video_path_stems = set(map(lambda t: str(Path(t).with_suffix('')), self.video_files))
        txt_path_stems = set(map(lambda t: str(Path(t).with_suffix('')), self.txt_files))
        self.path_stems = list(video_path_stems.intersection(txt_path_stems))
