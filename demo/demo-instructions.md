## Demo Instructions

A guide to running the demo as part of the presentation. Includes a sample model analysis utility with focus on TensorFlow, and PyTorch models.

> Tested on Ubuntu 22.04

### Clone Model Analyzer Demo Repository @GitHub:

`git clone https://github.com/mindboard-inc/tradewinds-model-analyzer.git`
`cd tradewinds-model-analyzer`

### Python Environment Setup

> Python Version Requirements

`Python 3.10+`
`pip 22.0.+`

> Recommended virtual environment setup with `venv`

`python -m venv env  # Create a virtual environment`
`source env/bin/activate  # Activate the virtual environment`

### Install Demo Dependencies:

`pip install -r requirements.demo.txt`

### Running Demo Script:

Run the sample script [demo-run.sh](./demo-run.sh) illustrating TensorFlow model analysis for the model:  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tf_model.h5?download=true.

`cd ./demo`
`./demo-run.sh`

> Downloading demo TensorFlow model
> --2024-10-15 11:32:21--  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tf_model.h5?download=true
> Resolving huggingface.co (huggingface.co)... 3.167.112.45, 3.167.112.96, 3.167.112.38, ...
> Connecting to huggingface.co (huggingface.co)|3.167.112.45|:443... connected.
>
> ...
>
> ../saved_models/tf_model.h5           100%[========================================================================>]  86.79M  35.8MB/s    in 2.4s    
> 2024-10-15 11:32:24 (35.8 MB/s) - ‘../saved_models/tf_model.h5’ saved [91005696/91005696]
>
> Running the demo model analyzer for tf_model.h5 model
> 2024-10-15 11:32:25,955 - INFO - Analyzing model: ../saved_models/tf_model.h5
> 2024-10-15 11:32:26,391 - INFO - Analysis results saved to ./analysis_results.2024-10-15_11-32-21.json
