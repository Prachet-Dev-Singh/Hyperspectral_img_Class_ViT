# download_data.py
import os
import requests
from config import DATA_PATH, INDIAN_PINES_URL_DATA, INDIAN_PINES_URL_GT
from utils import ensure_dir

def download_file(url, filename, data_path="."):
    full_path = os.path.join(data_path, filename)
    if os.path.exists(full_path):
        print(f"{filename} already exists at {full_path}. Skipping download.")
        return True

    ensure_dir(data_path)
    print(f"Downloading {filename} from {url} to {full_path}...")
    try:
        r = requests.get(url, stream=True, timeout=30) # Added timeout
        r.raise_for_status() # Raise an exception for HTTP errors
        with open(full_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        if os.path.exists(full_path): # Clean up partial download
            os.remove(full_path)
        return False

def download_indian_pines():
    print("Checking for Indian Pines dataset...")
    data_downloaded = download_file(INDIAN_PINES_URL_DATA, "Indian_pines_corrected.mat", DATA_PATH)
    gt_downloaded = download_file(INDIAN_PINES_URL_GT, "Indian_pines_gt.mat", DATA_PATH)

    if data_downloaded and gt_downloaded:
        print("Indian Pines dataset is ready.")
        return True
    else:
        print("Failed to obtain Indian Pines dataset. Please check URLs or download manually.")
        return False

if __name__ == "__main__":
    if download_indian_pines():
        print("Dataset available in:", os.path.abspath(DATA_PATH))
    else:
        print("Please ensure the dataset files are correctly placed in:", os.path.abspath(DATA_PATH))