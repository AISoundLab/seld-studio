'''
SELD STUDIO
Readme: feature_extraction.py

This file will iteratively take data points and labels from processed/train and procesed/test folders and extracts
Mel Spectrogram and MFCC features. The extracted features and labels are saved as .npy files under
processed/ folder.

Before running this file, make sure to run utils/data_chunkify.py

The folder sturucture before running this file must be:

root
...data
    ...processed
        ...train
            ...data
            ...labels
        ...test
            ...data
            ...labels
    ...train
        ...data
        ...labels
    ...test
        ...data
        ...labels
...utils
    ...data_chunkify.py

Libary dependencis:
1. numpy
2. librosa
3. csv
4. tqdm
5. warnings
6. torchaudio

Running Standalone: python feature_extraction.py
'''
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import os
import numpy as np
import librosa
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import csv

# Directories
os.chdir("..")
WORKING_DIR = os.getcwd()
DATA_DIR = os.path.join(WORKING_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
PROCESSED_TRAIN_DIR = os.path.join(PROCESSED_DIR, 'train')
PROCESSED_TEST_DIR = os.path.join(PROCESSED_DIR, 'test')
MODEL_DIR = os.path.join(WORKING_DIR, 'models', 'baseline')
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(MODEL_DIR, 'results')

# Create Directories
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)
if not os.path.exists(PROCESSED_TRAIN_DIR):
    os.makedirs(PROCESSED_TRAIN_DIR)
if not os.path.exists(os.path.join(PROCESSED_TRAIN_DIR, 'data')):
    os.makedirs(os.path.join(PROCESSED_TRAIN_DIR, 'data'))  
if not os.path.exists(os.path.join(PROCESSED_TRAIN_DIR, 'labels')):
    os.makedirs(os.path.join(PROCESSED_TRAIN_DIR, 'labels'))  
if not os.path.exists(PROCESSED_TEST_DIR):
    os.makedirs(PROCESSED_TEST_DIR)
if not os.path.exists(os.path.join(PROCESSED_TEST_DIR, 'data')):
    os.makedirs(os.path.join(PROCESSED_TEST_DIR, 'data'))  
if not os.path.exists(os.path.join(PROCESSED_TEST_DIR, 'labels')):
    os.makedirs(os.path.join(PROCESSED_TEST_DIR, 'labels'))  
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)    

# CONSTANTS
CLS_TO_IDX = {'Chink_and_clink':0,
               'Computer_keyboard':1,
               'Cupboard_open_or_close':2,
               'Drawer_open_or_close':3,
               'Female_speech_and_woman_speaking':4,
               'Finger_snapping':5,
               'Keys_jangling':6,
               'Knock':7,
               'Laughter':8,
               'Male_speech_and_man_speaking':9,
               'Printer':10,
               'Scissors':11,
               'Telephone':12,
               'Writing':13,
                'NOTHING': 14}

IDX_TO_CLS = {v: k for k, v in CLS_TO_IDX.items()}

def compute_mel_spectrogram(waveform, sr = 32000, n_fft = 2048, hop_length = 400, n_mels = 128, top_db = 80):

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=None,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )

    melspec = mel_spectrogram(waveform)[:,:,:-1]
    melspec=librosa.power_to_db(melspec,top_db=top_db)

    return melspec

def compute_mfcc(waveform, sr = 32000, n_fft = 2048, hop_length = 400, n_mels = 128, n_mfcc = 128):
    mfcc_transform = T.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={
        'n_fft': n_fft,
        'n_mels': n_mels,
        'hop_length': hop_length,
        'mel_scale': 'htk'
        }
    )

    mfcc = mfcc_transform(waveform)[:,:,:-1]
    mfcc=librosa.power_to_db(mfcc,top_db=80)
    return mfcc

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255* (spec_norm - spec_min + eps) / (spec_max - spec_min + eps)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

def data_preprocess(input_folder, processed_folder, key = None):
    # data
    print('Processing files in ', input_folder, ' ...')
    input_features = []
    target_labels = []
    data_path = os.path.join(input_folder, 'data')
    target_path_ = os.path.join(input_folder, 'labels')
    data = sorted(os.listdir(data_path))
    for sound in tqdm(data, leave=True, position = 0):
        target_name = 'label_' + sound.replace('_A', '').replace('.wav', '.csv')
        sound_path = os.path.join(data_path, sound)
        target_path = os.path.join(target_path_, target_name)
        samples, sr = torchaudio.backend.soundfile_backend.load(sound_path)
        melspec = compute_mel_spectrogram(samples)
        mfcc = compute_mfcc(samples)
        features = np.concatenate((melspec, mfcc), axis = 0)
        input_features.append(features)
        # label
        rows = []
        with open(target_path, 'r') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)
            # extracting each data row one by one
            for row in csvreader:
                rows.append(row)
            count = 0
            for i in range(len(rows)):
                if rows[i][1] == 'NOTHING':
                    count += 1
                    if count == 1:
                        temp1 = rows[i]
                else:
                    temp2 = rows[i]
            if count > 8:
                target_temp = temp1[1:]
            else:
                target_temp = temp2[1:]

            target_cls = [0] * len(CLS_TO_IDX)
            target_cls[CLS_TO_IDX[target_temp[0]]] = 1
            target_doa = [float(x) for x in target_temp[1:]]
            target = target_cls + target_doa
            target_labels.append(target)

    input_features = np.array(input_features)
    target_labels = np.array(target_labels)
    print('Saving data and labels to ', processed_folder, ' ...')
    np.save(os.path.join(processed_folder, key+'_input_features'), input_features)
    np.save(os.path.join(processed_folder, key+'_target_labels'), target_labels)

if __name__ == "__main__":
    data_preprocess(PROCESSED_TRAIN_DIR, PROCESSED_DIR, key = 'train')
    data_preprocess(PROCESSED_TEST_DIR, PROCESSED_DIR, key = 'test')

    