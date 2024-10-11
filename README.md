# vs-detection

This project implements a vocal sound classification system using various audio processing techniques and machine learning models. It aims to model four types of vocal sounds namely, `cough`, `laugh`, `mic tapping` and `people talking`, with focus on identifying a `cough`. The project contains independent audio samples, which are not part of the model training or validation. These are available under the folder `samples` and can be used as real-world examples. If you want to test a trained model on your own `.wav` files, you can simply place them in this folder and run inference, as explained in the last section.

## 1. Setup

This section guides you through setting up the necessary environment for the project.

1. Create and activate the Conda environment, from a `.yml` file, which will be called `vsdpy310`:

```bash
user123 % conda env create -f conda.yml
user123 % conda deactivate
user123 % conda activate vsdpy310
```

2. Install or update the IPython kernel: This step ensures that you have the correct IPython kernel installed, which is useful for running Jupyter notebooks, if you decide to use them for development or analysis.

```bash
user123 % conda install ipykernel --update-deps --force-reinstall
```

1. Create two directories: One directory is called `data`; this is where the training/validation data is expected to reside. The other directory is called `model`; this is where trained models and pertinent parameters are saved.

```bash
user123 % mkdir data
user123 % mkdir model
```

The folder structure for `data` is expected to be as follows:

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
│   └───test
│   │   │
│   │   │───coughing
│   │   │   cough2.wav
│   │   │
│   │   │───laugh
│   │   │   laugh2.wav
│   │   │
│   │   │───mic_tapping
│   │   │   tap2.wav
│   │   │
│   │   │───talking
│   │   │   talk2.wav
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

Available options for the `--method` argument are shown in the table below. For each option, we detail the corresponding configuration when turning sound into a 2D matrix of spectral components:

| Method     | Configuration                                              | Array Size   |
|------------|-----------------------------------------------------------|--------------|
| mels_big   | {"hop_length": 128, "n_fft": 2048, "n_mels": 224}         | (224, 251)   |
| mels_small | {"hop_length": 256, "n_fft": 2048, "n_mels": 128}         | (126, 128)   |
| mfcc_big   | {"hop_length": 256, "n_fft": 2048, "n_mfcc": 128}         | (126, 128)   |
| mfcc_small | {"hop_length": 512, "n_fft": 2048, "n_mfcc":  64}         | (63, 64)     |

The `--n_categories` argument can be set to either $2$ or $4$. If you choose $2$, then `laugh`, `mic_tapping` and `people_talking` are squeezed into a new category called `other`, otherwise for `n_categories`= $4$, we use the original labels.
It is recommended to use the default value for the sampling rate $sr=16000$.

## 3. Running Inference

To run inference on a trained model, you need first to activate the conda environment, then use the following command:

```bash
./run_inference.sh
```

This bash script will execute the `test_samples.py` python script, with the specified parameters. This script takes a list of audio samples, segments each of them into chunks of duration `CHUNK_LEN`, with a rolling window of $0.5$ seconds, and performs chunk-wise predictions. **This resembles real-life situations, where we continuously monitor the sounds and perform low-latency preodiction.**

One can customize the inference parameters by editing the `run_inference.sh` script. It uses default values for the parameters, which can be modified in the script if needed:

- CHUNK_LEN: 2
- SR: 16000
- N_CATEGORIES: 2
- METHOD: mfcc_small

**Important note #1**: You need to ensure that the files `train_model.sh` and `run_inference.sh` are executable. To do this manually, one can simply execute the following commands on the project root folder.

```bash
user123 % chmod +x train_model.sh
user123 % chmod +x run_inference.sh
```

**Important note #2**: When running inference on the provided samples, make sure that you use a combination for which you have trained a model!
