# vs-detection

This project implements a voice spoofing detection system using various audio processing techniques and machine learning models. It aims to distinguish between genuine and artificially generated voice samples to enhance security in voice-based authentication systems.

## Setup

This section guides you through setting up the necessary environment for the project.

1. Create and activate the Conda environment:

```bash
user123 ~% conda env create -f conda.yml
user123 ~% conda activate vsdpy310
```

2. Install or update the IPython kernel:

```bash
user123 ~% conda install ipykernel --update-deps --force-reinstall
```

This step ensures that you have the correct IPython kernel installed, which is useful for running Jupyter notebooks, if you decide to use them for development or analysis.

## Preprocessing

Save as figures, not csv since we can use iterators to better handle large amounts of data.

Note: folder `101846__stereodivo__vox_squad_laughing` was deleted because it needed special treatment

## Execution

To run the model training script, use the following command:

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

Available options for the `--method` argument are:
- mels_big
- mels_small
- mfcc_big
- mfcc_small

The `--n_categories` argument can be set to either $2$ or $4$.
It is recommended to use the default value for the sampling rate $sr=16000$. 