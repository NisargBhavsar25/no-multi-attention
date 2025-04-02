# BERT Transformer Implementation Research

This repository contains experimental implementations of different BERT transformer components and their performance analysis on the IMDB dataset.

## Project Structure

```
.
├── data/                  # Dataset storage
├── models/               # Different transformer implementations
│   ├── attention/       # Various attention mechanisms
│   ├── embeddings/      # Different embedding implementations
│   └── layers/          # Custom transformer layers
├── experiments/         # Experiment configurations and results
├── utils/              # Utility functions
└── notebooks/          # Jupyter notebooks for analysis
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

1. Train the model:
```bash
python train.py --config experiments/base_config.yaml
```

2. Evaluate performance:
```bash
python evaluate.py --model_path models/saved/your_model
```

## Metrics

The experiments track the following metrics:
- Accuracy on IMDB test set
- Training time
- Inference latency
- Model size
- Memory usage

## Results

Results are stored in the `experiments/results` directory, with detailed analysis in the notebooks. 