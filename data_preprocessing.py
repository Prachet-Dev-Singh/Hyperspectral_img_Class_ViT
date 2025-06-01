# data_preprocessing.py
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Using StandardScaler for PCA
import os

from config import (
    DATA_PATH, NUM_PCA_COMPONENTS, PATCH_SIZE,
    TEST_RATIO, VAL_RATIO_FROM_TRAIN, RANDOM_SEED
)
# Import NUM_CLASSES_ACTUAL to be able to set it
import config

def load_hsi_data(dataset_name="IndianPines"):
    if dataset_name == "IndianPines":
        data_file = os.path.join(DATA_PATH, "Indian_pines_corrected.mat")
        gt_file = os.path.join(DATA_PATH, "Indian_pines_gt.mat")
        
        if not os.path.exists(data_file) or not os.path.exists(gt_file):
            raise FileNotFoundError(
                f"Dataset files not found in {DATA_PATH}. "
                "Run download_data.py or place them manually."
            )
            
        data = loadmat(data_file)['indian_pines_corrected']
        gt = loadmat(gt_file)['indian_pines_gt']
        config.NUM_CLASSES_ACTUAL = np.max(gt) # Max class label before 0-indexing
        print(f"Original data shape: {data.shape}, GT shape: {gt.shape}")
        print(f"Number of classes (including background): {config.NUM_CLASSES_ACTUAL}")
        return data, gt
    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet.")

def apply_pca(data, num_components):
    h, w, c = data.shape
    data_reshaped = data.reshape(-1, c)
    
    # Scale before PCA for better results
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped.astype(np.float32)) # Ensure float32
    
    pca = PCA(n_components=num_components, random_state=RANDOM_SEED)
    data_pca = pca.fit_transform(data_scaled)
    
    print(f"PCA: Explained variance by {num_components} components: {np.sum(pca.explained_variance_ratio_):.4f}")
    return data_pca.reshape(h, w, num_components), pca, scaler

def create_patches(data_pca, gt, patch_size):
    h, w, c_pca = data_pca.shape
    pad_width = patch_size // 2
    
    # Pad the PCA data to handle border pixels
    # Ensure padding uses zeros or a value that won't affect standardized data negatively
    # For standardized data, padding with mean (0) is common.
    padded_data = np.pad(data_pca, 
                         ((pad_width, pad_width), (pad_width, pad_width), (0,0)), 
                         mode='constant', constant_values=0) # Pad with 0 for standardized data
    
    patches_list = []
    labels_list = []
    coordinates_list = [] # Store coordinates for consistent splitting

    for r_idx in range(h):
        for c_idx in range(w):
            label = gt[r_idx, c_idx]
            if label != 0: # Ignore background class (label 0)
                # Extract patch from padded_data using original (r_idx, c_idx) as center
                # The indices for padded_data will be (r_idx + pad_width, c_idx + pad_width) as center
                # Slice from (r_idx) to (r_idx + patch_size)
                patch = padded_data[r_idx : r_idx + patch_size, 
                                    c_idx : c_idx + patch_size, :]
                patches_list.append(patch)
                labels_list.append(label - 1) # Adjust labels to be 0-indexed
                coordinates_list.append((r_idx, c_idx))
                
    patches_arr = np.array(patches_list, dtype=np.float32)
    labels_arr = np.array(labels_list, dtype=np.int64)
    coordinates_arr = np.array(coordinates_list)

    # Update NUM_CLASSES_ACTUAL based on unique labels found (excluding background)
    config.NUM_CLASSES_ACTUAL = len(np.unique(labels_arr))
    print(f"Actual number of classes for model (0-indexed): {config.NUM_CLASSES_ACTUAL}")
    
    return patches_arr, labels_arr, coordinates_arr

def split_data(patches, labels, coordinates):
    num_samples = len(labels)
    indices = np.arange(num_samples)

    # Stratified split to maintain class proportions
    # Split pixels first based on their labels
    train_val_indices, test_indices, y_train_val_strat, _ = train_test_split(
        indices, labels, 
        test_size=TEST_RATIO, 
        random_state=RANDOM_SEED, 
        stratify=labels
    )
    
    # Further split train_val into train and validation
    # The test_size for this split is VAL_RATIO_FROM_TRAIN relative to the *original* dataset size,
    # but train_test_split's test_size is relative to the *input* dataset size (train_val_indices here).
    # So we need to adjust:
    # Effective val_ratio = VAL_RATIO_FROM_TRAIN / (1 - TEST_RATIO)
    if (1 - TEST_RATIO) == 0: # Avoid division by zero if TEST_RATIO is 1
        effective_val_ratio = 0 
    else:
        effective_val_ratio = VAL_RATIO_FROM_TRAIN / (1.0 - TEST_RATIO)
    
    if effective_val_ratio >= 1.0 or effective_val_ratio == 0: # Handle cases where validation set is too large or zero
        # If effective_val_ratio is >=1, it means VAL_RATIO_FROM_TRAIN >= (1-TEST_RATIO)
        # which implies val set would be all or more than the current train_val set.
        # This usually happens if TEST_RATIO is very high and VAL_RATIO_FROM_TRAIN is also significant.
        # Or if VAL_RATIO_FROM_TRAIN is 0.
        if VAL_RATIO_FROM_TRAIN == 0 or len(y_train_val_strat) == 0:
            train_indices = train_val_indices
            val_indices = np.array([], dtype=int) # Empty validation set
            print("Warning: Validation set is empty based on ratios.")
        else: # Give all to train if val_ratio is problematic, or minimal to val
             print(f"Warning: effective_val_ratio ({effective_val_ratio:.2f}) is problematic. Adjusting split.")
             # Heuristic: if train_val_indices is small, maybe just make a small val set.
             # This part might need careful adjustment based on expected dataset sizes.
             # For now, if problematic, let's default to a small validation split if possible.
             if len(train_val_indices) > 10 : # Arbitrary small number to allow split
                train_indices, val_indices, _, _ = train_test_split(
                    train_val_indices, y_train_val_strat,
                    test_size=min(0.1, effective_val_ratio), # Cap at 10% or calculated if smaller
                    random_state=RANDOM_SEED,
                    stratify=y_train_val_strat
                )
             else: # Not enough to split, all to train
                train_indices = train_val_indices
                val_indices = np.array([], dtype=int)
                print("Warning: Not enough samples in train_val for validation split. Val set is empty.")
    else:
        train_indices, val_indices, _, _ = train_test_split(
            train_val_indices, y_train_val_strat,
            test_size=effective_val_ratio,
            random_state=RANDOM_SEED,
            stratify=y_train_val_strat
        )

    X_train, y_train = patches[train_indices], labels[train_indices]
    coords_train = coordinates[train_indices]
    
    X_val, y_val = patches[val_indices], labels[val_indices]
    coords_val = coordinates[val_indices]
    
    X_test, y_test = patches[test_indices], labels[test_indices]
    coords_test = coordinates[test_indices]
    
    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return (X_train, y_train, coords_train), \
           (X_val, y_val, coords_val), \
           (X_test, y_test, coords_test)


def get_prepared_data():
    """Main function to orchestrate data loading and preprocessing."""
    hsi_data, gt = load_hsi_data(dataset_name=config.DATASET_NAME)
    data_pca, _, _ = apply_pca(hsi_data, NUM_PCA_COMPONENTS)
    
    # Patches are (N, patch_size, patch_size, num_pca_components)
    all_patches, all_labels, all_coords = create_patches(data_pca, gt, PATCH_SIZE)
    
    (X_train, y_train, _), (X_val, y_val, _), (X_test, y_test, _) = split_data(
        all_patches, all_labels, all_coords
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    # Test the preprocessing steps
    from utils import set_seeds
    from download_data import download_indian_pines

    set_seeds(RANDOM_SEED)
    if not download_indian_pines():
        exit("Dataset not available. Exiting.")

    print("\n--- Testing Data Preprocessing ---")
    X_train, y_train, X_val, y_val, X_test, y_test = get_prepared_data()
    
    print(f"\nShape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    if X_val.shape[0] > 0:
        print(f"Shape of X_val: {X_val.shape}, y_val: {y_val.shape}")
    else:
        print("X_val is empty.")
    print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"Number of classes for model: {config.NUM_CLASSES_ACTUAL}")
    print(f"Min/Max label in y_train: {np.min(y_train)}, {np.max(y_train)}")
    
    # Verify patch content (optional)
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 3)
    # axes[0].imshow(X_train[0, :, :, 0]) # Show first PCA component of first patch
    # axes[0].set_title(f"Train Patch 0, Label {y_train[0]}")
    # axes[1].imshow(X_train[0, :, :, X_train.shape[-1]//2])
    # axes[2].imshow(X_train[0, :, :, -1])
    # plt.show()