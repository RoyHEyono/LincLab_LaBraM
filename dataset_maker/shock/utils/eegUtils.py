import h5py
import mne
import numpy as np


standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

def preprocessing_cnt(cntFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200):
    # reading cnt
    raw = mne.io.read_raw_cnt(cntFilePath, preload=True, data_format='int32')
    raw.drop_channels(['M1', 'M2', 'VEO', 'HEO'])
    if 'ECG' in raw.ch_names:
        raw.drop_channels(['ECG'])

    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
    raw = raw.notch_filter(50.0)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)
    eegData = raw.get_data(units='uV')

    return eegData, raw.ch_names

def preprocessing_fif(npyFilePath):
    # reading cnt
    raw = mne.io.read_raw_fif(npyFilePath, preload=True)
    # raw.drop_channels(['M1', 'M2', 'VEO', 'HEO'])
    # if 'ECG' in raw.ch_names:
    #     raw.drop_channels(['ECG'])

    for ch_n in raw.ch_names:
        if ch_n.upper() not in standard_1020:
            raw.drop_channels([ch_n]) 
    
    eegData = raw.get_data(units='uV')

    return eegData, raw.ch_names


def preprocessing_edf(edfFilePath, l_freq=0.1, h_freq=75.0, sfreq:int=200, drop_channels: list=None, standard_channels: list=None):
    # reading edf
    raw = mne.io.read_raw_edf(edfFilePath, preload=True)
    if drop_channels is not None:
        useless_chs = []
        for ch in drop_channels:
            if ch in raw.ch_names:
                useless_chs.append(ch)
        raw.drop_channels(useless_chs)

    if standard_channels is not None and len(standard_channels) == len(raw.ch_names):
        try:
            raw.reorder_channels(standard_channels)
        except:
            return None, ['a']

    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
    raw = raw.notch_filter(50.0)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=5)
    eegData = raw.get_data(units='uV')

    return eegData, raw.ch_names

 
def readh5(h5filePath):
    with h5py.File('matrix.h5', 'r', libver='latest', swmr=True) as f:
        dset = f['data']
        shape = dset.shape
        dtype = dset.dtype

        if dset.chunks:
            np_array = np.empty(shape, dtype=dtype)
            dset.read_direct(np_array)
        else: 
            np_array = dset[()]
    return np_array


if __name__ == '__main__':
    print(readh5('./').shape)
