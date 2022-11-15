# -*- coding:utf-8 -*-
import lmdb
import csv

#------------------------------------------ video-audio-text write into lmdb over!-------------------------------------

class lmdb_handle():
    def __init__(self,db_name):
        self.env = lmdb.open(db_name,map_size=int(1e13),lock=False)
        # self.txn =  self.env.begin(write=True)

        if not self.env:
            raise IOError('Cannot open lmdb dataset', db_name)
        #try:
        #  with self.env.begin(write=False) as txn:
        #       self.cnt = int(txn.get('num'.encode('utf-8')).decode('utf-8'))
        #except:
        #    self.cnt=0
        #print('current samples:', self.cnt)
    def stat(self):
        return(self.txn.stat())

    def get(self,k):
        with self.env.begin(write=False) as txn:
            return txn.get(k)
        
        # return bytes(self.txn.get(k)) if self.txn.get(k) else None

    def add_tensors(self,cache):
        assert isinstance(cache,dict)
        with self.env.begin(write=True) as txn:
            for k, v in cache.items():
                if not txn.get(k.encode()):
                    #if txn.get(k.encode()) is not None:continue   
                    txn.put(k.encode(), v.numpy())
                else:
                    print('{} is exist!'.format(k))

    def close(self):
        self.env.close()
    def set_mapsize(self,size):
        self.env.set_mapsize(size)
        
if __name__ =="__main__":
    lmdb1 = lmdb.open('db_name1')
    