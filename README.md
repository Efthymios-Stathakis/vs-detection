# vs-detection

This project implements a vocal sound classification system using various audio processing techniques and machine learning models. It aims to distinguish between four types of vocal sounds namely, `cough`, `laugh`, `mic tapping` and `people talking`, with focus on `cough`.

## 1. Setup

This section guides you through setting up the necessary environment for the project.

1. Create and activate the Conda environment:

```bash
user123 ~% conda env create -f conda.yml
user123 ~% conda activate vsdpy310
```

2. Install or update the IPython kernel: This step ensures that you have the correct IPython kernel installed, which is useful for running Jupyter notebooks, if you decide to use them for development or analysis.

```bash
user123 ~% conda install ipykernel --update-deps --force-reinstall
```

3. Create a directory called `data`, where the training data are expected to be found.

```bash
user123 ~% mkdir data
```

The folder structure is expected to be as follows:

```
data 
└───original_audio
│   │
│   └───train
│   │   │
│   │   │───coughing
│   │   │   cough1.wav
│   │   │
│   │   │───laugh
│   │   │   laugh1.wav
│   │   │
│   │   │───mic_tapping
│   │   │   tap1.wav
│   │   │
│   │   │───talking
│   │   │   talk1.wav
│   │   │   
│   │   │   ...
│   └───train
│   │   │
│   │   │───coughing
│   │   │   cough1.wav
│   │   │
│   │   │───laugh
│   │   │   laugh1.wav
│   │   │
│   │   │───mic_tapping
│   │   │   tap1.wav
│   │   │
│   │   │───talking
│   │   │   talk1.wav
│   │
└───augmented_audio
│   │
│   └───train
│   │   │   ...
```

## 2. Model Training

To run the model training script, you need first to activate the conda environment, then use the following command:

```bash
./train_model.sh
```

You can customize the training parameters by editing the `train_model.sh` script or by running the Python script directly with specific arguments:

```bash
python entrypoint.py \
      --chunk_len 2 \
      --verbose 0 \
      --sr 16000 \
      --n_categories 2 
      --method mfcc_big
```

Available options for the `--method` argument are shown in the table below. For each option, we detail the corresponding configuration when turning sound into 2D matrix of spectral components:

| Method     | Configuration                                              | Array Size   |
|------------|-----------------------------------------------------------|--------------|
| mels_big   | {"hop_length": 128, "n_fft": 2048, "n_mels": 224}         | (63, 64)     |
| mels_small | {"hop_length": 256, "n_fft": 2048, "n_mels": 128}         | (126, 128)   |
| mfcc_big   | {"hop_length": 256, "n_fft": 2048, "n_mfcc": 128}         | (126, 128)   |
| mfcc_small | {"hop_length": 512, "n_fft": 2048, "n_mfcc":  64}         | (224, 251)   |

The `--n_categories` argument can be set to either $2$ or $4$.
It is recommended to use the default value for the sampling rate $sr=16000$.

## 3. Running Inference

To run inference on a trained model, you need first to activate the conda environment, then use the following command:

```bash
./run_inference.sh
```

This bash script will execute the `test_samples.py` python script, with the specified parameters. This script takes a list of audio samples, segments each of them into chunks of duration `CHUNK_LEN` and performs chunk-wise predictions. **This resembles real-life situations, where we continuously monitor the sounds and perform low-latency preodiction.**

One can customize the inference parameters by editing the `run_inference.sh` script. It uses default values for the parameters, which can be modified in the script if needed:

- CHUNK_LEN: 2
- SR: 16000
- N_CATEGORIES: 2
- METHOD=mfcc_small

Important note: You need to ensure that the files `train_model.sh` and `run_inference.sh` are executable. To do this manually, one can simply execute the following commands on the project root folder.

```bash
user123 ~% chmod +x train_model.sh
user123 ~% chmod +x run_inference.sh
```
