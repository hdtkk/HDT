# HDT: Hierarchical Discrete Transformer for Multivariate Time Series Forecasting


## Installation

### 1. Dataset Installation
Run the following command to download and preprocess the dataset (Taxi as an example) in each training and inference stage:

```python
dataset = get_dataset("taxi_30min", regenerate=True)  # Set regenerate=True for the first time
```

## Training

### Stage 1 Training
```python
python ./stage1_downsampled_target/stage1_dowsample_run.py  # Discrete downsampled target training
python ./stage1_target/stage1_run.py   # Discrete target training
```

### Stage 2 Training
```python
python ./Stage2_downsampled_generation/main.py  # Discrete downsampled target generation
python ./Stage2_target_generation/main.py   # Discrete target generation
```
## Eval
### Inference
```python
python ./Stage2_inference/inference.py  # Target forecasting
```

## Citing

To cite this repository:

```tex
@software{pytorchgithub,
    author = {Kashif Rasul},
    title = {{P}yTorch{TS}},
    url = {https://github.com/zalandoresearch/pytorch-ts},
    version = {0.6.x},
    year = {2021},
}
```

