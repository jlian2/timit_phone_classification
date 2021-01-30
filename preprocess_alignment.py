phoneme_61_39 = {
    'ao': 'aa',  # 1
    'ax': 'ah',  # 2
    'ax-h': 'ah',
    'axr': 'er',  # 3
    'hv': 'hh',  # 4
    'ix': 'ih',  # 5
    'el': 'l',  # 6
    'em': 'm',  # 6
    'en': 'n',  # 7
    'nx': 'n',
    'eng': 'ng',  # 8
    'zh': 'sh',  # 9
    "ux": "uw",  # 10
    "pcl": "sil",  # 11
    "tcl": "sil",
    "kcl": "sil",
    "qcl": "sil",
    "bcl": "sil",
    "dcl": "sil",
    "gcl": "sil",
    "h#": "sil",
    "#h": "sil",
    "pau": "sil",
    "epi": "sil",
    "q": "sil",
}

phoneme39_list = [
    'iy', 'ih', 'eh', 'ae', 'ah', 'uw', 'uh', 'aa', 'ey', 'ay', 'oy', 'aw', 'ow',  # 13 phns
    'l', 'r', 'y', 'w', 'er', 'm', 'n', 'ng', 'ch', 'jh', 'dh', 'b', 'd', 'dx',  # 14 phns
    'g', 'p', 't', 'k', 'z', 'v', 'f', 'th', 's', 'sh', 'hh', 'sil'  # 12 pns
]
phoneme61_list = [
    'iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr',
    'ax-h', 'jh',
    'ch', 'b', 'd', 'g', 'p', 't', 'k', 'dx', 's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng', 'em', 'nx',
    'en', 'eng', 'l', 'r', 'w', 'y', 'hh', 'hv', 'el', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'q', 'pau', 'epi',
    'h#',
]
import math

def convert39():
	phoneme39 = dict(zip(phoneme39_list, list(range(0,39))))
	phoneme61 = dict()
	for i in phoneme61_list:
		if i in phoneme39_list:
			phoneme61[i] = phoneme39[i]
		else:
			phoneme61[i] = phoneme39[phoneme_61_39[i]]

	id2phoneme = dict()
	for k, v in phoneme61.items():
		if v not in id2phoneme:
			id2phoneme[v] = [k]
		else:
			id2phoneme[v].append(k)

	#print(phoneme61,'\n')

	return phoneme61

def to_framewise(wav, phoneme61):
	File = open(wav).readlines()
	pt = 200	# sample point
	ans = ""	# 'sil' for the 1st frame
#	assert 400 < int(File[0].strip('\n').split(' ')[1]) # so that the 1st frame must be 'sil'
	last = int(File[-1].strip('\n').split(' ')[1])

	enter = False
	for i in range(len(File)):
		line = File[i]
		line = line.strip('\n').split(' ')
#		print(line)
		h_window=200 #win_length * 0.5
		hop_length=160
		start_time=int(line[0])
		end_time=int(line[1])
		start_time = (start_time - h_window) if start_time >= h_window else 0
		end_time = (end_time - h_window) if end_time >= h_window else 0
		times = (end_time // hop_length) - (start_time // hop_length)  + (1 if start_time % hop_length == 0 else 0) - (1 if end_time % hop_length == 0 else 0)
		ph = str(phoneme61[line[2]])+" "
		ans += ph*int(times)
	
	ans = ans[:-1]
	return ans

import os
from glob import glob
def aligned_timit(filedir,out):
	phoneme61 = convert39()
	ID_list = filedir.split('/')
	ID = ID_list[-3]+'-'+ID_list[-2]+'-'+ID_list[-1]+' '
	ans = to_framewise(filedir+'.PHN', phoneme61)
	print(ID + ' '+ ans)
	ans = ID + ans + '\n'
	out.write(ans)

def main():
	outfile = 'converted_aligned_phones.txt'
	with open(outfile, 'w') as out:
		# test-split
		test_root = 'TIMIT/TEST'
		test_phone_file = [line.rstrip() for line in open('test_split.txt')]
		#print(test_phone_file)
		for line in test_phone_file:
			filedir = line.split('-')
			filedir = os.path.join(test_root,filedir[0],filedir[1],filedir[2])
			aligned_timit(filedir,out)
		# train-split
		train_root = 'TIMIT/TRAIN'
		train_phone_file = [line.rstrip() for line in open('train_split.txt')]
		#print(train_phone_file)
		for line in train_phone_file:
			filedir = line.split('-')
			filedir = os.path.join(train_root,filedir[0],filedir[1],filedir[2])
			aligned_timit(filedir,out)
	out.close()
if __name__ == '__main__':
	main()
