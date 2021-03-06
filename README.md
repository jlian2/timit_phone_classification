# timit_phone_classification
### Preprocess timit dataset
#### TIMIT-corpus (total 6300 files)
  -- TRAIN (4620 files)<br>
  -- TEST (1680 files)<br>
* download timit and place TIMIT/ inside this directory
#### 3 files
* train_split.txt: list training data files
* test_split.txt: list testing data files
* converted_aligned_phones.txt: framewise label
#### Features & Settings
* feature type:
  * fbank
  * mfcc
* settings:
  * sample_rate = 16000
  * num_mels = 80 
  * num_mfcc = 13 
  * window_size = 25 ms
  * stride = 10 ms
* convert to framewise:
  * for each wav file, match length between input features and labels:<br>
    * if input length is longer then truncate the extra final frames
    * if label length is longer then duplicate the last frame to match the label length
  (those frames are usually silence)
  * for each wav file, remove silence frames at the beginning and end<br>
  * after processing each file, we get (seq_length,feature_dim) for each file<br>
  concat all the frames to a list, so we get<br>
  `data.shape=(total_frames,feature_dim)` and `label.shape=(total_frames,)`
  * save processed training/testing data and labels to .npy<br>
  
#### Run
extract features for each file and convert to framewise training/testing data
* fbank(default: delta=False, delta_delta=False, apply_cmvn=True)<br>
`python preprocess_data.py -f=fbank --extract`<br>
`feature dim = 80`
* mfcc(default: delta=True, delta_delta=True, apply_cmvn=True)<br>
`python preprocess_data.py -f=mfcc --extract --delta=True --delta_delta=True`<br>
`feature dim = 39`

** update: add neighbor frames for training<br>
input `-n=k` -> total `2k+1` frames (concat k frames on both sides and predict the center frame)
* fbank(default: n=5, delta=False, delta_delta=False, apply_cmvn=True)<br>
`python preprocess_data_stack.py -f=fbank -n=k`<br>
`feature dim = 80*(2k+1)`
* mfcc(default: n=5, delta=True, delta_delta=True, apply_cmvn=True)<br>
`python preprocess_data_stack.py -f=mfcc -n=k --delta=True --delta_delta=True`<br>
`feature dim = 39*(2k+1)`
