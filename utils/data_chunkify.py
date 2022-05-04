'''
SELD STUDIO
Readme: data_chunkify.py

This file will iteratively take a data point from L3DAS 2022 dataset and converts 
each 30 second audio clip to 30 one second clips, and creates 30 csv files, 
each having the label for the 30 clips. The train audio clips will be saved under 
'data/processed/stereo_train/data' and the train labels will be saved under 
'data/processed/stereo_train/labels'. 

Before running this script: You must download the 'train' and 'test' folders using 
our 'data_download' script. 

The directory stucture BEFORE running this script should be:

root
...data
    ...train
        ...data
        ...labels
    ...test
        ...data
        ...labels
...utils
    ...data_chunkify.py

The directory stucture AFTER running this script will be:

root
...data
    ...processed
        ...stereo_train
            ...data
            ...labels
        ...stereo_test
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
3. pandas
4. tqdm
5. scipy.io
6. warnings

Running Standalone: python data_chunkify.py
'''
import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
import scipy.io.wavfile
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Directories
WORKING_DIR = os.path.join('..',os.getcwd())
DATA_DIR = os.path.join(WORKING_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
STEREO_TRAIN_DIR = os.path.join(PROCESSED_DIR, 'stereo_train')
STEREO_TEST_DIR = os.path.join(PROCESSED_DIR, 'stereo_test')
MODEL_DIR = os.path.join(WORKING_DIR, 'models', 'baseline')
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(MODEL_DIR, 'results')

# Create Directories
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)
if not os.path.exists(STEREO_TRAIN_DIR):
    os.makedirs(STEREO_TRAIN_DIR)
if not os.path.exists(os.path.join(STEREO_TRAIN_DIR, 'data')):
    os.makedirs(os.path.join(STEREO_TRAIN_DIR, 'data'))  
if not os.path.exists(os.path.join(STEREO_TRAIN_DIR, 'labels')):
    os.makedirs(os.path.join(STEREO_TRAIN_DIR, 'labels'))  
if not os.path.exists(STEREO_TEST_DIR):
    os.makedirs(STEREO_TEST_DIR)
if not os.path.exists(os.path.join(STEREO_TEST_DIR, 'data')):
    os.makedirs(os.path.join(STEREO_TEST_DIR, 'data'))  
if not os.path.exists(os.path.join(STEREO_TEST_DIR, 'labels')):
    os.makedirs(os.path.join(STEREO_TEST_DIR, 'labels'))  
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
# Considering sound files with only one active sound per frame
OV_SUBSETS = ['ov1']
# audio file time
FILE_DURATION = 30
# audio sampling frequency
SAMPLING_FREQUENCY = 32000
# audio chunks duration
DURATION = 1.0
# Output generated every 100 ms
STEP = 0.1
# Value for normalization for tanh
MAX_LOC_VALUE = 2.
# Number of 100 ms frames in a second
NUM_FRAMES = 10

# Save Audio Chunks
def save_audio_chunck(audio, path, sr=SAMPLING_FREQUENCY):
    audio=audio.transpose()
    scipy.io.wavfile.write(path, sr, audio)

# Extract labels chunks
def extract_labels_from_csv(target_path, target_name, processed_folder):
    tot_steps =int(FILE_DURATION/STEP)
    num_classes = len(CLS_TO_IDX)
    num_frames = int(FILE_DURATION/STEP)
    cl = np.zeros((tot_steps, num_classes))
    loc = np.zeros((tot_steps, num_classes, 3))
    quantize = lambda x: round(float(x) / STEP) * STEP
    get_frame = lambda x: int(np.interp(x, (0,FILE_DURATION),(0,num_frames-1)))
    df = pd.read_csv(target_path)
    active_frames = []
    for index, s in df.iterrows():
        start = quantize(s['Start'])
        end = quantize(s['End'])
        start_frame = get_frame(start)
        end_frame = get_frame(end)
        class_id = CLS_TO_IDX[s['Class']]
        sound_frames = np.arange(start_frame, end_frame+1)
        active_frames += list(sound_frames)
        for f in sound_frames:
            cl[f][class_id] = 1.
            loc[f][class_id][0] = s['X']
            loc[f][class_id][1] = s['Y']
            loc[f][class_id][2] = s['Z']
    loc = loc / MAX_LOC_VALUE    
    all_frames = list(np.arange(0, tot_steps))
    active_frames = set(active_frames)
    empty_frames = [x for x in all_frames if x not in active_frames]
    for f in empty_frames:
        cl[f][14] = 1.

    target_dict = {}
    for i in range(len(loc)):
        cls_idx = np.argmax(cl[i])
        target_dict[i] = [i, IDX_TO_CLS[cls_idx], loc[i][cls_idx][0], loc[i][cls_idx][1], loc[i][cls_idx][2]]
    df = pd.DataFrame.from_dict(target_dict, orient='index', columns=['FRAME', 'CLASS','X', 'Y', 'Z'])
    df = df.set_index('FRAME')  
    count = 0
    for i in range(0, FILE_DURATION * 10, 10):
        df_temp = df.iloc[i:i+10]
        filepath = os.path.join(processed_folder, 'labels', target_name[:-4] + '_chunk'+str(count)+'.csv')
        count += 1
        if count > FILE_DURATION:
            count = 0
        df_temp.to_csv(filepath)

    cl = cl.reshape(FILE_DURATION, 10, -1)
    loc = loc.reshape(FILE_DURATION, 10, -1)

# Extract audio chunks
def make_stereo(input_folder, processed_folder):
    data_path = os.path.join(input_folder, 'data')
    target_path = os.path.join(input_folder, 'labels')
    print ('Processing files in ' + input_folder + ' ...')
    data = sorted(os.listdir(data_path))
    # Consider files from only mic A
    data = [i for i in data if i.split('.')[0].split('_')[-1]=='A']
    
    for idx, sound in enumerate(tqdm(data, leave=True, position=0)):
        ov_set = sound.split('_')[-3]
        if ov_set in OV_SUBSETS:
            sound_path = os.path.join(data_path, sound)
            target_name = 'label_' + sound.replace('_A', '').replace('.wav', '.csv')
            target_path = os.path.join(data_path, target_name)
            #replace data with labels
            target_path = '\\'.join((target_path.split('\\')[:-2] + ['labels'] + [target_path.split('\\')[-1]])) 
            for j in range(FILE_DURATION):
                samples, sr = librosa.load(sound_path, sr = SAMPLING_FREQUENCY, mono=False, offset = j, duration = DURATION)
                #stereo = convert_multichannel_to_stereo(samples)
                save_audio_chunck(samples, os.path.join(processed_folder, 'data', sound[:-4] +'_chunck{}.wav'.format(j)), sr=SAMPLING_FREQUENCY)
        extract_labels_from_csv(target_path, target_name, processed_folder)

if __name__ == "__main__":
    make_stereo(TRAIN_DIR, STEREO_TRAIN_DIR)
    make_stereo(TEST_DIR, STEREO_TEST_DIR)
