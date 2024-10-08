import numpy as np 
import pandas as pd
from pathlib import Path
from scipy.io import wavfile


def wave_to_csv(filepath):
    """
    Extract data from a wave file and add timestamps
    """
    samplerate, data = wavfile.read(filepath)
    data = data.reshape(len(data), -1)
    # print(len(data)/samplerate)
    if (samplerate != 16000):print(f"Sample rate is {samplerate}")
    t = np.arange(start=0, stop=(len(data)/ samplerate)  , step= 1/ samplerate)
    t = t[:len(data)]
    col = ["Time (seconds)"] + ["CH{}".format(i) for i in range(data.shape[1])]
    
    return pd.DataFrame(data=np.hstack((t.reshape(-1,1), data)), columns=col)


def convert_track_to_label(raw_data, raw_label):
    """
    Convert label track into a numpy vector
    Args:
        raw_data (pd.dataframe): data with timestamp
        raw_label (pd.dataframe): label 

    Returns:
        label (np.array): label vector
    """

    raw_data_time = raw_data.values[:, 0]
    label_track = raw_label.values

    label_vector = np.zeros(len(raw_data_time))

    for i in range(label_track.shape[0]):
        start_index = np.argmax(raw_data_time >= label_track[i, 0])
        if label_track[i, 1] + label_track[i, 0] <= raw_data_time[-1]:
            # in case some label length exceeds the actual data length
            end_index = np.argmax(raw_data_time >= label_track[i, 1] + label_track[i, 0])
        else:
            end_index = -1

        label_vector[start_index:end_index] = 1
    
    return label_vector
