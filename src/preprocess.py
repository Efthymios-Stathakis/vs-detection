from pathlib import Path
from .utils import search_for_data_files, get_features
from typing import Any, List, Dict, Tuple

map_dirs = {
    "coughing_batch_2" : "cough",
    "coughing"         : "cough",
    "people_talking"   : "talking",
    "mic_tapping"      : "tapping",
    "laugh"            : "laughing"
}

def preprocess(
    input_dir_list: List[str],
    params: Dict[str, int],
    sr: int = 16000,
    chunk_len: int = 2,
    verbose: int = 0
) -> Tuple[List[Any], List[str]]:
    """
    Preprocess audio files from multiple directories and extract features.

    This function takes a list of input directories, processes the audio files
    within each directory, extracts features using the provided parameters,
    and returns lists of features and corresponding labels.

    Args:
        input_dir_list (List[str]): List of paths to input directories containing audio files.
        params (Dict[str, int]): Parameters for feature extraction (e.g., mel spectrogram or MFCC).
        sr (int, optional): Sampling rate for audio processing. Defaults to 16000.
        chunk_len (int, optional): Length of audio chunks in seconds. Defaults to 2.
        verbose (int, optional): Verbosity level for logging. Defaults to 0.

    Returns:
        Tuple[List[Any], List[str]]: Two lists containing:
            - features_list (List[Any]): Extracted features for all processed audio files.
            - labels_list (List[str]): Corresponding labels for the extracted features.

    Note:
        The function expects WAV files and corresponding label files in each input directory.
        It uses the `map_dirs` dictionary to map directory names to label names.
    """
    
    features_list = []
    labels_list   = []
    for input_dir in input_dir_list:
        wav_files = search_for_data_files(Path(input_dir), '*.wav')
        lbl_files = search_for_data_files(Path(input_dir), '*.label')
        label = map_dirs[input_dir.split('/')[-1]]
        
        mels_list = get_features(wav_files, 
                                 lbl_files, 
                                 sr=sr,
                                 chunk_len=chunk_len,
                                 params=params, 
                                 verbose=verbose)
        features_list.extend(mels_list)
        labels_list.extend([label]*len(mels_list))
    
    return features_list, labels_list