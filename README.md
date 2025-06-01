Create virtual env: python3 -m venv hsi_env\n
activate it: source hsi_env/bin/activate\n

install required libraries:\n
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Or your CUDA version, or cpuonly\n
pip install numpy scipy scikit-learn requests matplotlib # Added matplotlib for potential visualization\n
pip install torchinfo # Optional, for model summary\n
pip install tqdm\n

then after running the file for downloading the dataset, the final file to run is the main.py
