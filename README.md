Create virtual env: python3 -m venv hsi_env
activate it: source hsi_env/bin/activate

install required libraries:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Or your CUDA version, or cpuonly
pip install numpy scipy scikit-learn requests matplotlib # Added matplotlib for potential visualization
pip install torchinfo # Optional, for model summary
pip install tqdm

then after running the file for downloading the dataset, the final file to run is the main.py
