#!/bin/bash

# Default values (matching those in argparse)
ENTRYPOINT="python test_samples.py"
CHUNK_LEN=2
SR=16000
N_CATEGORIES=2
METHOD="mfcc_small"

# Function to display usage
usage() {
    echo "Usage: $0 <keyword>"
    echo "Available keywords:"
    echo "mels_big", "mels_small", "mfcc_big", "mfcc_small"
    exit 1
}

# Check if a keyword is provided
# if [ $# -eq 0 ]; then
#     usage
# fi

# Execute the entrypoint with the selected parameters
$ENTRYPOINT \
    --chunk_len $CHUNK_LEN \
    --sr $SR \
    --n_categories $N_CATEGORIES \
    --method $METHOD
