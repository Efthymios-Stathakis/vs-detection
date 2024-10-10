import argparse, json, os
import librosa
import logging
import numpy as np
import tensorflow as tf
from src.utils import create_mels, create_mfcc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mels_big   = {"hop_length": 128, "n_fft": 2048, "n_mels": 224}
mels_small = {"hop_length": 256, "n_fft": 2048, "n_mels": 128}
mfcc_big   = {'hop_length': 256, "n_fft": 2048, "n_mfcc": 128}
mfcc_small = {'hop_length': 512, "n_fft": 2048, "n_mfcc": 64}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio inference script")
    parser.add_argument("--chunk_len", type=int, default=2, help="Chunk length (default: 2)")
    parser.add_argument("--sr",        type=int, default=None, help="Sample rate (default: None)")
    parser.add_argument("--method", choices=["mels_big", "mels_small", "mfcc_big", "mfcc_small"], 
                        default="mfcc_big", help="Feature extraction method (default: mfcc_big)")
    parser.add_argument("--n_categories", type=int, choices=[2, 4], default=2, help="Number of categories (default: 2)")

    args = parser.parse_args()
    
    # Update variables with parsed arguments
    chunk_len    = args.chunk_len
    sr           = args.sr
    n_categories = args.n_categories
    method       = args.method
        
    # Update params based on the chosen method
    if method == "mels_big":
        params = mels_big
    elif method == "mels_small":
        params = mels_small
    elif method == "mfcc_big":
        params = mfcc_big
    else:  # mfcc_small
        params = mfcc_small
    
    # Initialize parameters based on the numbrer of categories
    if n_categories == 4:
        categories = ["tapping", "talking", "laughing", "cough"]
        activation = "softmax"
        loss_fn    = "categorical_crossentropy"
    elif n_categories == 2:
        categories = ["other", "cough"]
        activation = "sigmoid"
        loss_fn    = "binary_crossentropy"
        
    logger.info(f"Starting audio inference with parameters: chunk_len={chunk_len}, sr={sr}, method={method}, n_categories={n_categories}")
    
    json_path = f"./model/{method}_{n_categories}_n_cat_params.json"
    logger.info(f"Loading model parameters from {json_path}")
    with open(json_path, "r") as json_params:
        train_params = json.load(json_params)
        X_train_max = train_params["X_train_max"]
        X_train_min = train_params["X_train_min"]
    
    model_path = f"./model/{method}_{n_categories}_n_cat_cnn_model"
    logger.info(f"Loading model from {model_path}")
    loaded_model = tf.keras.models.load_model(model_path)

    wav_samples = [filename for filename in os.listdir("./samples/") if "wav" in filename]
    logger.info(f"Found {len(wav_samples)} WAV files for processing")

    for audio_sample in wav_samples:
        logger.info(f"Processing audio sample: {audio_sample}")
        y, sr = librosa.load(f"./samples/{audio_sample}", sr=16000)
        placeholders = np.arange(0, len(y)- chunk_len*sr, 0.5*sr, dtype=int)
        logger.info(f"Analyzing {len(placeholders)} chunks for this sample")

        for x1 in placeholders:
            x2 = int(x1+chunk_len*sr)
            audio_signal = y[x1:x2]
            if "n_mels" in params: 
                x = create_mels(audio_signal, sr, **params)
            elif "n_mfcc" in params: 
                x = create_mfcc(audio_signal, sr, **params)
            x = np.expand_dims(x, axis=-1)
            x = np.expand_dims(x, axis=0)
            x = (x-X_train_min) / (X_train_max - X_train_min)
            predictions = loaded_model.predict(x)
            
            logger.info(f'Chunk {x1/sr:.2f}s - {x2/sr:.2f}s predictions:')
            for i, label in enumerate(categories):
                logger.info(f'  {label}: {predictions[0][i]:.4f}')

    logger.info("Audio inference completed successfully")