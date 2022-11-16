import csv 
from utils.lmdb_class import lmdb_handle
import argparse
import os
from os.path import join as opj
import lmdb
def get_args_parser():
    parser = argparse.ArgumentParser('VAT pre-training', add_help=False)
    parser.add_argument('--lmdb_merge_path',default='lmdb_you_0_5_merge',type=str,help='')
    parser.add_argument('--lmdb_rank_pro_folder',default='lmdb_you_0_5',type=str,help='')
    parser.add_argument('--merge_csv',default='lmdb_you_0_5_merge.csv',type=str,help='')
    return parser


def merge_lmdb_csv(args,lmdb_path_list):
    assert args.merge_csv, 'args.merge_csv should not be None!'

    success_ = 0
    fail_ = 0 
    # csv-write,写入多行用writerows
    for lmdb_path1 in lmdb_path_list:
        env = lmdb.open(lmdb_path1,map_size=1e12,max_readers=4096,readonly=True,lock=False,meminit=False,readahead=False)
        # read lmdb
        with env.begin(write=False) as txn:
            # assert txn.stat()['entries']%3==0, 'lmdb is not tri-pairs!'
            database = txn.cursor()

            with open(args.merge_csv,"w") as csvfile:
                writer = csv.writer(csvfile) 
                writer.writerow(["audio","text","video"])

                for idx,(key,value) in enumerate(database):
                    key = key.decode()
                    try:
                        if key.startswith('video'):
                            id_video = key
                            id_audio = key.replace('video','audio')
                            audio_tokens = txn.get(id_audio.encode())
                            assert audio_tokens is not None, '{} is not in lmdb!!!'.format(id_audio)
                            id_text = key.replace('video','text')
                    
                            text_tokens = txn.get(id_text.encode())
                            assert text_tokens is not None, '{} is not in lmdb!!!'.format(id_text)

                            writer.writerow([id_audio,id_text,id_video])
                            if success_ %1000==1:
                                print('success_:{}'.format(success_),[id_audio,id_text,id_video])
                            success_ += 1
                        else:
                            pass       
                    except Exception as e:
                        fail_ += 1
                        print(e)
        env.close()

    print('result_csv entries is {},fail:{}'.format(success_,fail_))

def get_lmdb_path_list(folder1):
    lmdb_path_list = [ opj(folder1,lmdb_name1)  for lmdb_name1 in os.listdir(folder1) ]
    return lmdb_path_list


def main(args):
    lmdb_path_list = get_lmdb_path_list(args.lmdb_rank_pro_folder)
    merge_lmdb_csv(args,lmdb_path_list)



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
    
