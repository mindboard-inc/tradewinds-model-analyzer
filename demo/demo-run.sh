#!/bin/bash
SCRIPTS=../scripts
MODEL_PATH=../saved_models/tf_model.h5
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPATH=./analysis_results.$TIMESTAMP.json


# Download the TF model to saved_models
if [ -f $MODEL_PATH ]
then
    echo "Model already exists, skipping download"
else
    echo "Downloading demo TensorFlow model"
    wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tf_model.h5?download=true -O $MODEL_PATH
fi


# Run the demo
echo "Running the demo model analyzer for tf_model.h5 model"
python $SCRIPTS/local-model-analyzer.py $MODEL_PATH --output $OUTPATH

