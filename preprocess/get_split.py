import os
import sys
from glob import glob

def get_train_spilt():
    timit_root = 'TIMIT/TRAIN/'
    train_set = sorted(glob(timit_root + "/*/*/*.WAV"))
    with open('TIMIT/train_split.txt', 'w') as f:
        for item in train_set:
            item = item.split('/')
            name = item[4].split('.')[0]
            f.write(item[2]+'-'+item[3]+'-'+name+'\n')
def get_test_split():
    timit_root = 'TIMIT/TEST/'
    test_set = sorted(glob(timit_root + "/*/*/*.WAV"))
    with open('TIMIT/test_split.txt', 'w') as f:
        for item in test_set:
            item = item.split('/')
            name = item[4].split('.')[0]
            f.write(item[2]+'-'+item[3]+'-'+name+'\n')
            #f.write(item[2]+'-'+item[3]+'\n')
def main():
    get_train_spilt()
    get_test_split()
if __name__ == '__main__':
    main()
