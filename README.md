# 🌈 Hyperspectral Image Classification using Vision Transformers (ViT)

This repository provides an implementation for classifying hyperspectral images using Vision Transformers (ViT). The pipeline includes dataset preprocessing, model training, and evaluation.

## 🛠️ Setup Instructions

### 1. Create and Activate a Virtual Environment

```bash
python3 -m venv hsi_env
source hsi_env/bin/activate
```

### 2. Install Required Dependencies

#### a. Install PyTorch (choose your appropriate CUDA version)

```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If you're using CPU only, use:

```bash
pip install torch torchvision torchaudio
```

#### b. Install additional Python libraries

```bash
pip install numpy scipy scikit-learn requests matplotlib tqdm torchinfo
```

## 📥 Download Dataset

Download the required dataset by running:

```bash
python download_data.py
```

## 🚀 Run the Model

After downloading the dataset, execute the main training and evaluation pipeline:

```bash
python main.py
```

## 📁 Project Structure

| File                   | Description                                   |
|------------------------|-----------------------------------------------|
| `config.py`            | Configuration and hyperparameters             |
| `data_preprocessing.py`| Data normalization and preprocessing          |
| `download_data.py`     | Script to download the hyperspectral dataset  |
| `hsi_dataset.py`       | Custom PyTorch dataset class                  |
| `losses.py`            | Custom loss functions                         |
| `main.py`              | Entry point for training and evaluation       |
| `model.py`             | Vision Transformer model definition           |
| `train_eval.py`        | Training and evaluation logic                 |
| `utils.py`             | Utility functions                             |

## 📝 Notes

- Ensure the virtual environment is activated before running any script.
- Adjust the CUDA version in the PyTorch installation command based on your system.
- `torchinfo` is optional and used for displaying model architecture summaries.
- `matplotlib` is included for potential visualizations.

