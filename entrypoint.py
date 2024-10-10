import argparse, json, os
import numpy as np
import pandas as pd
import logging

from src.preprocess import preprocess
from src.model import create_model
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.legacy import Adam 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mels_big   = {"hop_length": 128, "n_fft": 2048, "n_mels": 224}
mels_small = {"hop_length": 256, "n_fft": 2048, "n_mels": 128}
mfcc_big   = {'hop_length': 256, "n_fft": 2048, "n_mfcc": 128}
mfcc_small = {'hop_length': 512, "n_fft": 2048, "n_mfcc": 64}
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio processing script")
    parser.add_argument("--chunk_len", type=int, default=2, help="Chunk length (default: 2)")
    parser.add_argument("--verbose",   type=int, default=0, help="Verbosity level (default: 0)")
    parser.add_argument("--sr",        type=int, default=None, help="Sample rate (default: None)")
    parser.add_argument("--method", choices=["mels_big", "mels_small", "mfcc_big", "mfcc_small"], 
                        default="mfcc_big", help="Feature extraction method (default: mfcc_big)")
    parser.add_argument("--n_categories", type=int, choices=[2, 4], default=2, help="Number of categories (default: 2)")

    args = parser.parse_args()

    # Update global variables with parsed arguments
    chunk_len    = args.chunk_len
    verbose      = args.verbose
    sr           = args.sr
    n_categories = args.n_categories
    method       = args.method
    
    logger.info(f"Starting script with parameters: chunk_len={chunk_len}, verbose={verbose}, sr={sr}, n_categories={n_categories}, method={method}")

    # Update params based on the chosen method
    if method == "mels_big":
        params = mels_big
    elif method == "mels_small":
        params = mels_small
    elif method == "mfcc_big":
        params = mfcc_big
    else:  # mfcc_small
        params = mfcc_small
    
    logger.info(f"Using {method} method with params: {params}")

    if n_categories == 4:
        categories = ["tapping", "talking", "laughing", "cough"]
        map_cat    = { "tapping":0, "talking":1, "laughing":2, "cough":3}
        activation = "softmax"
        loss_fn    = "categorical_crossentropy"
    elif n_categories == 2:
        categories = ["other", "cough"]
        map_cat    = {"laughing":0, "talking":0, "tapping":0, "cough":1}
        activation = "sigmoid"
        loss_fn    = "binary_crossentropy"
        
    logger.info(f"Using {n_categories} categories: {categories}")
    logger.info(f"Last hidden layer activation: {activation}, Loss function: {loss_fn}")

    base_dir_train = "./data/original_audio/train"
    base_dir_test  = "./data/original_audio/test"
    train_input_dirs  = [os.path.join(base_dir_train, dir)       
                        for dir in os.listdir(base_dir_train) if "DS_Store" not in dir] 
    test_input_dirs   = [os.path.join(base_dir_test, dir) 
                        for dir in os.listdir(base_dir_test) if "DS_Store" not in dir] 

    logger.info("Starting preprocessing stage")
    (train_features, train_labels) = preprocess(train_input_dirs, params)
    (test_features,  test_labels)  = preprocess(test_input_dirs , params)
    logger.info("Preprocessing completed")

    X_train = np.array(train_features)
    y_train = np.array([map_cat[label] for label in train_labels])

    X_test = np.array(test_features)
    y_test = np.array([map_cat[label] for label in test_labels])

    logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    logger.info("Normalizing data")
    X_train_norm = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_test_norm  = (X_test  - X_train.min()) / (X_train.max() - X_train.min())

    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    logger.info("Creating and compiling model")
    model = create_model(X_train_norm.shape[1:], 
                         n_categories=n_categories, 
                         activation=activation)
    model.compile(optimizer=Adam(), loss=loss_fn, metrics=['accuracy'])
    
    logger.info("Starting model training")
    hist = model.fit(X_train_norm, y_train_encoded, 
                 validation_data=(X_test_norm, y_test_encoded), 
                 batch_size=16, 
                 epochs=25)
    
    logger.info("Model training completed")

    # Save the model in SavedModel format
    model_path = f"./model/{method}_{n_categories}_n_cat_cnn_model"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Optionally, you can also save the model weights separately
    # weights_path = f"./model/{method}_{n_categories}_n_cat_cnn_weights.h5"
    # model.save_weights(weights_path)
    # logger.info(f"Model weights saved to {weights_path}")
   
    # Save the model and preprocessing paraemters
    train_params = {
        "input_shape" : X_train_norm.shape[1:], 
        "n_categories": n_categories,
        "activation"  : activation,
        "X_train_max" : X_train.max(),
        "X_train_min" : X_train.min()
        }
    json_path    = f"./model/{method}_{n_categories}_n_cat_params.json"
    with open(json_path, 'w') as f:
        json.dump(train_params, f )
    logger.info(f"Training parameters saved to {json_path}")


    y_predicted = model.predict(X_test_norm)
    mat = confusion_matrix(y_test_encoded.argmax(axis=1), 
                           y_predicted.argmax(axis=1))
    logger.info(f"Confusion matrix is {mat}")
    
    mat = confusion_matrix(y_test_encoded.argmax(axis=1), 
                           y_predicted.argmax(axis=1), 
                           normalize="true")
    logger.info(f"Normalized confusion matrix is {mat}")
    
    logger.info("Script execution completed")

