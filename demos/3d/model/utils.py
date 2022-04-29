import numpy as np
import torch
from scipy.signal import stft

'''
Miscellaneous utilities
'''
def load_model(model, optimizer, path, cuda):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # load state dict of wrapped module
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    try:
        model.load_state_dict(torch.load(checkpoint['model_state_dict'],
                                    map_location=lambda storage, location: storage),
                                    strict=False)
    except:
        # work-around for loading checkpoints where DataParallel was saved instead of inner module
        from collections import OrderedDict
        model_state_dict_fixed = OrderedDict()
        prefix = 'module.'
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith(prefix):
                k = k[len(prefix):]
            model_state_dict_fixed[k] = v
        model.load_state_dict(model_state_dict_fixed)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'state' in checkpoint:
        state = checkpoint['state']
    else:
        # older checkpoints only store step, rest of state won't be there
        state = {'step': checkpoint['step']}
    return state


def spectrum_fast(x, nperseg=512, noverlap=128, window='hamming', cut_dc=True,
                  output_phase=True, cut_last_timeframe=True):
    '''
    Compute magnitude spectra from monophonic signal
    '''

    f, t, seg_stft = stft(x,
                        window=window,
                        nperseg=nperseg,
                        noverlap=noverlap)

    output = np.abs(seg_stft)

    if output_phase:
        phase = np.angle(seg_stft)
        output = np.concatenate((output,phase), axis=-3)

    if cut_dc:
        output = output[:,1:,:]

    if cut_last_timeframe:
        output = output[:,:,:-1]

    return output

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

def predictions_list(sed, doa, max_loc_value=2.,num_frames=600, num_classes=14, max_overlaps=3):
    '''
    Process sed and doa output matrices (model's output) and generate a list of active sounds
    and their location for every frame. The list has the correct format for the Challenge results
    submission.
    '''
    predictions = []
    for i, (c, l) in enumerate(zip(sed, doa)):  #iterate all time frames
        c = np.round(c)  #turn to 0/1 the class predictions with threshold 0.5
        l = l * max_loc_value  #turn back locations between -2,2 as in the original dataset
        l = l.reshape(num_classes, max_overlaps, 3)  #num_class, event number, coordinates
        if np.sum(c) == 0:  
             predictions.append([i, IDX_TO_CLS[14], 0, 0, 0]) 
        else:
            for j, e in enumerate(c):  #iterate all events
                if e != 0:  #if an avent is predicted
                    #append list to output: [time_frame, sound_class, x, y, z]
                    predicted_class = int(j/max_overlaps)
                    predicted_class_label = IDX_TO_CLS[predicted_class]
                    num_event = int(j%max_overlaps)
                    curr_list = [i, predicted_class_label, l[predicted_class][num_event][0], l[predicted_class][num_event][1], l[predicted_class][num_event][2]]

                    predictions.append(curr_list)

    return predictions