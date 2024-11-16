from pathlib import Path
from shock.utils import h5Dataset
from shock.utils.eegUtils import preprocessing_fif,preprocessing_edf
import random

random.seed(42)
all_alexmi = list(range(1,9)) 
all_physionetmi = list(range(1,110))

random.shuffle(all_alexmi)
random.shuffle(all_physionetmi )

def process_alexmi_filename(x):
  out=[]
  for num in x:
    out=out+["subject"+str(num)+".raw.fif"]
  return out
alexmi_pretrain = all_alexmi[:len(all_alexmi) // 2]
alexmi_pretrain = process_alexmi_filename(alexmi_pretrain)
alexmi_finetune = all_alexmi[len(all_alexmi) // 2:]

def process_physionetmi_filename(x):
  out=[]
  suffix=['04', '06', '08', '10', '12', '14']
  for num in x:
    for even in suffix:
      out=out+[f"S{str(num).zfill(3)}R{even}.edf"]
  return out
physionetmi_pretrain = all_physionetmi[:len(all_physionetmi) // 2]
physionetmi_pretrain = process_physionetmi_filename(physionetmi_pretrain)
physionetmi_finetune = all_physionetmi[len(all_physionetmi) // 2:]

savePath = Path('/network/scratch/q/qingchen.hu/eeg_processed')
rawDataPath = Path('/network/scratch/q/qingchen.hu/mne_data/MNE-alexeeg-data/record/806023/files/')
group = [file for file in rawDataPath.glob('*.fif') if file.name in alexmi_pretrain]

# preprocessing parameters
RESAMPLING_RATE = 200  # Hz
FMIN = 8  # Hz
FMAX = 32  # Hz
# rsfreq = RESAMPLING_RATE
# channel number * rsfreq
chunks = (16, RESAMPLING_RATE)
log_file = Path('/home/mila/q/qingchen.hu/LincLab_LaBraM/checkpoints/hdf5_processing_log.txt')
def log_name(file_name):
    with open(log_file, "a") as log:
        log.write(file_name + "\n")

name='alexeeg_selected'
dataset = h5Dataset(savePath, name)
log_name(f'-----------PROCESSING H5DATASET: {name}-----------')

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

    log_name(fifFile.name)

dataset.save()

# for .edf files
rawDataPath = Path('/network/scratch/q/qingchen.hu/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/')
group = [file for file in rawDataPath.glob('**/*.edf') if file.name in physionetmi_pretrain]
chunks = (64, RESAMPLING_RATE)

name='physionetMI_selected'
dataset = h5Dataset(savePath, name)
log_name(f'-----------PROCESSING H5DATASET: {name}-----------')
for edfFile in group:
    print(f'processing {edfFile.name}')
    eegData, chOrder = preprocessing_edf(edfFile)
    chOrder = [s.upper() for s in chOrder]
    # eegData = eegData[:, :-10*rsfreq]
    grp = dataset.addGroup(grpName=edfFile.stem)
    dset = dataset.addDataset(grp, 'eeg', eegData, chunks)

    # dataset attributes
    dataset.addAttributes(dset, 'lFreq', FMIN)
    dataset.addAttributes(dset, 'hFreq', FMAX)
    dataset.addAttributes(dset, 'rsFreq', RESAMPLING_RATE)
    dataset.addAttributes(dset, 'chOrder', chOrder)

    log_name(edfFile.name)

dataset.save()