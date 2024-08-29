from pathlib import Path
from shock.utils import h5Dataset
from shock.utils.eegUtils import preprocessing_fif

savePath = Path('/network/scratch/r/roy.eyono/eeg_processed')
rawDataPath = Path('/network/scratch/r/roy.eyono/raw/MNE-alexeeg-data/record/806023/files')
group = rawDataPath.glob('*.fif')

# preprocessing parameters
RESAMPLING_RATE = 200  # Hz
FMIN = 8  # Hz
FMAX = 32  # Hz

# channel number * rsfreq
chunks = (16, RESAMPLING_RATE)

dataset = h5Dataset(savePath, 'alexeeg')
for fifFile in group:
    print(f'processing {fifFile.name}')
    eegData, chOrder = preprocessing_fif(fifFile)
    chOrder = [s.upper() for s in chOrder]
    # eegData = eegData[:, :-10*rsfreq]
    grp = dataset.addGroup(grpName=fifFile.stem)
    dset = dataset.addDataset(grp, 'eeg', eegData, chunks)

    # dataset attributes
    dataset.addAttributes(dset, 'lFreq', FMIN)
    dataset.addAttributes(dset, 'hFreq', FMAX)
    dataset.addAttributes(dset, 'rsFreq', RESAMPLING_RATE)
    dataset.addAttributes(dset, 'chOrder', chOrder)

dataset.save()
