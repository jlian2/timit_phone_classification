import os
import sys
import random
import numpy as np
import librosa
import argparse
from argparse import Namespace

############
# SETTINGS #
############
sample_rate = 16000
num_mels = 80 # int, dimension of feature
num_mfcc = 13 # int, number of MFCCs
window_size = 25 # int, window size for FFT (ms)
stride = 10 # int, window stride for FFT

def extract_feature(input_file, feature='fbank', delta=False, delta_delta=False, cmvn=True, save_dir=None):
    y, sr = librosa.load(input_file, sr=sample_rate)

    if feature == 'fbank':
        ws = int(sr*0.001*window_size)
        st = int(sr*0.001*stride)
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=num_mels, n_fft=ws, hop_length=st)
        feat = np.log(feat + 1e-6) # log-scaled
    elif feature == 'mfcc':
        ws = int(sr*0.001*window_size)
        st = int(sr*0.001*stride)
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc, n_fft=ws, hop_length=st)
        feat[0] = librosa.feature.rms(y, hop_length=st, frame_length=ws)
    # add delta
    feat = [feat]
    if delta:
        feat.append(librosa.feature.delta(feat[0]))
    if delta_delta:
        feat.append(librosa.feature.delta(feat[0], order=2))
    feat = np.concatenate(feat, axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:,np.newaxis]) / (feat.std(axis=1)+1e-16)[:,np.newaxis]
    if save_dir is not None:
        out = np.swapaxes(feat, 0, 1).astype('float32')
        np.save(save_dir,out)

def preprocess_split(args,split,split_list,in_root):
    print('preprocessing {0}...'.format(split))
    count = 0
    for line in split_list:
        count += 1
        infile_name = '/'.join(line.split('-'))
        wav = os.path.join(in_root,infile_name)
        save_root = 'features_'+args.feature
        output_dir = os.path.join(save_root,split)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if args.feature == 'fbank':
            extract_feature(wav+'.WAV', args.feature, args.delta, args.delta_delta, args.apply_cmvn, os.path.join(output_dir,line+'.npy'))
        elif args.feature == 'mfcc':
            extract_feature(wav+'.WAV', args.feature, args.delta, args.delta_delta, args.apply_cmvn, os.path.join(output_dir,line+'.npy'))
        else:
            raise ValueError('Unsupported Acoustic Feature: ' + args.feature)