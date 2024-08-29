import warnings
import os
import argparse
import numpy as np

import moabb
from moabb.paradigms import MotorImagery
from moabb.datasets.utils import find_intersecting_channels
import mne

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

moabb.set_log_level("info")

if __name__ == "__main__":

    # EVENTS are: 
    #     "left_hand"
    #     "right_hand"
    #     "feet"
    #     "navigation"
    #     "subtraction"
    #     "word_ass"

    EVENTS_MAPPING = {
        "left_hand": 0,
        "right_hand": 1,
        "feet": 2,
    }

    RESAMPLING_RATE = 200  # Hz
    FMIN = 8  # Hz
    FMAX = 32  # Hz
    CHANNELS = []
    PARADIGM = MotorImagery(
        events=["left_hand", "right_hand", "feet"],
        n_classes=2,  # minimum number of classes
        # fmin=FMIN,
        # fmax=FMAX,
        # channels=CHANNELS,
        resample=RESAMPLING_RATE,
    )
    DATASETS = PARADIGM.datasets
    DATASETS_SIG = {
        "AlexandreMotorImagery": "ALEXEEG",
        "BNCI2014-001": "BNCI",
        "BNCI2014-002": "BNCI",
        "BNCI2014-004": "BNCI",
        "BNCI2015-001": "BNCI",
        "BNCI2015-004": "BNCI",
        # "Cho2017": "GIGADB",
        "GrosseWentrup2009": "MUNICHMI",
        # "Lee2019-MI": "Lee2019-MI",
        "PhysionetMotorImagery":"EEGBCI",
        # "Schirrmeister2017": "SCHIRRMEISTER2017",
        # "Shin2017A": "BBCIFNIRS",  # "BBCI EEG-fNIRS"
        "Weibo2014": "WEIBO",
        "Zhou2016": "ZHOU",
    }

    # Use argparse to extract two arguments from the command line:
    # input_dir and output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="./raw", help="Output directory"
    )
    args = parser.parse_args()

    def mapp_labels_to_int(labels):
        return np.array([EVENTS_MAPPING[label] for label in labels]).astype(int)
    
    for dataset in DATASETS:
        if PARADIGM.is_valid(dataset):
            print(f'Downloading dataset {dataset.code}')
            truncated_file_data = True
            truncated_file_labels = True

            while truncated_file_data or truncated_file_labels:
                try:
                    # assign output_dir path to env. variable for dataset 
                    signifier = DATASETS_SIG[dataset.code]
                    os.environ[f'MNE_DATASETS_{signifier}_PATH'] = args.output_dir

                    # load data
                    subj_list = dataset.subject_list
                    
                    # if len(subj_list) > 10:
                    #     subj_list = subj_list[:10]
                    
                    for subj in subj_list:
                        X, labels, metadata = PARADIGM.get_data(
                            dataset=dataset, 
                            subjects=[subj],
                        )

                        # map str labels to int
                        labels = mapp_labels_to_int(labels)

                        # save data
                        np.save(
                            os.path.join(
                                args.output_dir, f'{dataset.code}_{subj}_x.npy'
                            ),
                            X
                        )
                        np.save(
                            os.path.join(
                                args.output_dir, f'{dataset.code}_{subj}_y.npy'
                            ),
                            labels
                        )
                        metadata.to_csv(
                            os.path.join(
                                args.output_dir, f'{dataset.code}_{subj}_meta.csv'
                            )
                        )

                        truncated_file_data = False
                        truncated_file_labels = False
                    
                except KeyError as error:
                    truncated_file_data = False
                    truncated_file_labels = False
                    print("\tAn exception occured. ", error)
                
                except OSError:
                    os.remove(
                        os.path.join(
                            args.output_dir, f'{dataset.code}_{subj}_x.npy'
                        )
                    )
                    os.remove(
                        os.path.join(
                            args.output_dir, f'{dataset.code}_{subj}_y.npy'
                        )
                    )
                    print("\tTruncated file, re-downloading")