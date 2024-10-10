import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from librosa.feature import melspectrogram as mels
from librosa.feature import mfcc

def create_mels(
    audio_signal, 
    sr:int,
    hop_length:int = 128,
    n_mels:int = 224,
    n_fft:int = 2048,
):
    # ms_arr = mels(y=audio_signal, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)
    # return librosa.power_to_db(ms_arr, ref=np.max)
    
    return mels(y=audio_signal, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)

def create_mfcc(
    audio_signal, 
    sr:int,
    hop_length:int = 256,
    n_mfcc:int = 128,
    n_fft:int = 2048,
):
    mfcc_arr = mfcc(y=audio_signal, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mfcc=n_mfcc)
    return np.swapaxes(mfcc_arr, 0, 1)
    
def save_spectrogram(input_arr, sr:int, image_file:str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    librosa.display.specshow(input_arr, sr=sr)
    fig.savefig(image_file)
    plt.close(fig)

def get_features(
    audio_files,
    label_files, 
    sr : int = 16000,
    chunk_len : int  = 2,
    params = {}, 
    output_dir : str = None, 
    verbose : int = 0):
    """ This module takes in an audio and extracts Mel spectrograms or Mel Frequency
    components. We deal with two cases:
    1. Cough audio files: These files contain annotated cough segments, which are 
    explicitly extracted. If the chunk duration is smaller that `chunk_len` then we 
    add random amount of silence on each ends, such that the total duration equals 
    `chunk_len`. 
    
    2. For audio files that are not cough, there is no silence so we chunk them 
    directly into segments of length `chunk_len`, but we skip `chunk_len` seconds 
    to avoid high correlation between contiguous segments.

    Args:
        audio_files (list[str]): List of audio (.wav) files
        label_files (list[str]): List of files with annotation
        output_dir (str): Output path for each spectrogram
        sr (int, optional): Audio sample rate, defaults to 16000.
        chunk_len (int, optional): Duration of extracted segments. Defaults to 2.
        verbose (int, optional): Prints out information => Defaults to 0.
    """
    
    np.random.seed(48)
    feature_list = []
    for audio_file, label_file in zip(audio_files, label_files):
        
        if verbose: print(audio_file, " || ", label_file)
        basename = '_'.join(str(audio_file).split('/')[1:-1])
        label_df = pd.read_csv(label_file)
        duration_col = [col for col in label_df if "length" in col.lower()][0]
        
        y, sr = librosa.load(audio_file, sr=sr)
        if len(label_df[duration_col].values.tolist()) == 0:
            start_times = np.arange(0, len(y) - chunk_len*sr, chunk_len*sr) // sr
            durations   = [chunk_len] * len(start_times)
        else:
            start_times = label_df["Time(Seconds)"].values.tolist()
            durations   = label_df[duration_col].values.tolist()
        for idx, (start_time, duration) in enumerate(zip(start_times, durations)):
            end_time  = start_time + min(duration, chunk_len)
            start_idx = int(np.floor(start_time * sr))
            end_idx   = int(np.ceil( end_time   * sr))
            filename = basename + f"_{idx}"
            if verbose: 
                print(f"{filename}: Start {start_time:.2f} || end {end_time:.2f} || sr {sr}")
            
            audio_signal = y[start_idx:end_idx]
            if len(audio_signal) < (chunk_len*sr):
                diff      = chunk_len*sr - len(audio_signal)
                pad_left  = np.random.randint(1, diff)
                pad_right = diff - pad_left
                audio_signal = np.concatenate(
                    (np.zeros(pad_left,), 
                     audio_signal, 
                     np.zeros(pad_right,))) 
                
            # Check the dicionary inputs to figure out if we want Mel 
            # spectrogram or if we want the Mel frequency coefficients
            if "n_mels" in params: 
                spectral_arr = create_mels(audio_signal, sr, **params)
            elif "n_mfcc" in params: 
                spectral_arr = create_mfcc(audio_signal, sr, **params)
            feature_list.append(np.expand_dims(spectral_arr, -1))
            
            if output_dir:
                if not os.path.exists(output_dir): os.makedirs(output_dir)
                output_path = os.path.join(output_dir, filename)
                save_spectrogram(spectral_arr, sr, output_path)
    
    return feature_list

def search_for_data_files(search_dir: Path, data_file_wildcard: str):
    """
    Searches recursively for files named according to data_file_wildcard
    """
    data_files = list(search_dir.rglob(data_file_wildcard))
    if not data_files:
        print(f"Warning: Could not find any data files named: {data_file_wildcard}")
    
    return data_files