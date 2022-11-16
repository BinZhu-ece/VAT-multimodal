import csv 
from utils.lmdb_class import lmdb_handle
import argparse
import os
from os.path import join as opj
import lmdb
def get_args_parser():
    parser = argparse.ArgumentParser('VAT pre-training', add_help=False)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--lmdb_rank_pro_folder',default='lmdb_you_0_5',type=str,help='')
    parser.add_argument('--lmdb_merge_path',default='lmdb_you_0_5_merge',type=str,help='')
    parser.add_argument('--csv_path',default=None,type=str,help='')
    return parser


def merge_lmdb(lmdb_path_list,result_lmdb_path):

    env_result = lmdb.open(result_lmdb_path,map_size=1e12)
    txn_result = env_result.begin(write=True)
    print('Before merge: result_lmdb stat is {}'.format(txn_result.stat()))

    success_ = 0
    fail_ = 0  
    for lmdb_path1 in lmdb_path_list:
        env = lmdb.open(lmdb_path1,map_size=1e8,max_readers=4096,readonly=True,lock=False,meminit=False,readahead=False)
        with env.begin(write=False) as txn:
            print('lmdb:{},stat:{}'.format(lmdb_path1,txn.stat()))
            database = txn.cursor()
            for idx,(key,value) in enumerate(database):
                try:
                    txn_result.put(key,value)
                    success_ += 1
                    if success_ % 1000 == 1:
                        txn_result.commit()
                        print('commit once, success_:{}'.format(success_))
                        txn_result = env_result.begin(write=True) 
                except Exception as e:
                    fail_ += 1
                    print(e)
    print('resule_lmdb stat is {}'.format(env_result.stat()))
    env_result.close()

def get_lmdb_path_list(folder1):
    lmdb_path_list = [   opj(folder1,lmdb_name1)  for lmdb_name1 in os.listdir(folder1) ]
    return lmdb_path_list


def main(args):

    lmdb_path_list = get_lmdb_path_list(args.lmdb_rank_pro_folder)

    merge_lmdb(lmdb_path_list,args.lmdb_merge_path)



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
    
