import os
import sys
import random
import numpy as np
import pandas as pd
from utility import *

label = dict()
phone_file = [line.rstrip() for line in open('converted_aligned_phones.txt')]
for f in phone_file:
    filename = f.split(' ')[0]
    label[filename] = np.array(f.split(' ')[1:])

def match_length(inputs, labels):
    input_len, label_len = inputs.shape[0], labels.shape[0]
    if input_len > label_len:
        inputs = inputs[:label_len,:]
    elif input_len < label_len:
        pad_val = inputs[-1,:]
        pad_len = label_len - input_len
        inputs = np.concatenate((inputs,[pad_val]*pad_len),axis=0)
    assert(inputs.shape[0]==labels.shape[0])
    return inputs, labels

def toframe(feat_dir,split_list):
    split = feat_dir.split('/')[1]
    print('converting {0} data to framewise...'.format(split))
    frame = []
    phone_label = []
    count = 0
    for fileid in split_list:
        count += 1 
        print('[{0}] {1}'.format(count,fileid))
        origin_feat = np.load(os.path.join(feat_dir,fileid+'.npy'))
        features, labels= match_length(origin_feat,label[fileid])
        sil_count_f = 0
        while int(labels[sil_count_f]) == 38:
            sil_count_f += 1
        sil_count_b = -1
        while int(labels[sil_count_b]) == 38:
            sil_count_b -= 1
        phone_label = phone_label + labels[sil_count_f:features.shape[0]+sil_count_b+1].tolist()
        for i in range(sil_count_f,features.shape[0]+sil_count_b+1):
            frame.append(features[i].tolist()) 
    print('------------ finished ------------')
    print('total frames: {0}'.format(len(frame)))
    print('feature dim: {0}'.format(len(frame[0])))
    print('phone labels: {0}'.format(len(phone_label)))
    print('----------------------------------')
    #df = pd.DataFrame(frame, columns=range(1, len(frame[0])+1))
    #df.to_csv('{}.csv'.format(feat_dir.split('/')[1]))
    np.save('{0}.npy'.format(split.lower()),frame)
    np.save('{0}_label.npy'.format(split.lower()),phone_label)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature', choices=['fbank', 'mfcc'])
    parser.add_argument('--extract', action='store_true', help='extract features if not extracted yet')
    parser.add_argument('--delta', default=False, type=bool, help='Append Delta', required=False)
    parser.add_argument('--delta_delta', default=False, type=bool, help='Append Delta Delta', required=False)
    parser.add_argument('--apply_cmvn', default=True, type=bool, help='Apply CMVN on feature', required=False)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    train_root = 'TIMIT/TRAIN'
    test_root = 'TIMIT/TEST'
    train_list = [line.rstrip() for line in open('train_split.txt')]
    test_list = [line.rstrip() for line in open('test_split.txt')]
    ### extract features of each .WAV ###
    if args.extract:
        preprocess_split(args,'TRAIN',train_list,train_root)
        preprocess_split(args,'TEST',test_list,test_root)
    ### convert to framewise dataset ###
    feat_train_root = 'features_{0}/TRAIN'.format(args.feature)
    feat_test_root = 'features_{0}/TEST'.format(args.feature)
    toframe(feat_train_root,train_list)
    toframe(feat_test_root,test_list)

if __name__ == '__main__':
    main()