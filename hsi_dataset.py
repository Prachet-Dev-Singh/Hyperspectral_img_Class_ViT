# hsi_dataset.py
import torch
from torch.utils.data import Dataset

class HSIDataset(Dataset):
    def __init__(self, patches, labels, transform=None):
        """
        Args:
            patches (numpy.ndarray): Array of patches, shape (N, H, W, C_pca).
            labels (numpy.ndarray): Array of labels, shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.patches = patches
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch = self.patches[idx] # (H, W, C_pca)
        label = self.labels[idx]

        # Convert to PyTorch tensor
        # ViT expects input as (C, H, W)
        patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1) # (C_pca, H, W)
        label_tensor = torch.tensor(label).long()

        if self.transform:
            patch_tensor = self.transform(patch_tensor)
            
        return patch_tensor, label_tensor

if __name__ == "__main__":
    # Test HSIDataset
    from data_preprocessing import get_prepared_data
    from utils import set_seeds
    from download_data import download_indian_pines
    from config import RANDOM_SEED, BATCH_SIZE
    from torch.utils.data import DataLoader

    set_seeds(RANDOM_SEED)
    if not download_indian_pines():
        exit("Dataset not available. Exiting.")

    print("\n--- Testing HSIDataset and DataLoader ---")
    X_train, y_train, X_val, y_val, X_test, y_test = get_prepared_data()

    train_dataset = HSIDataset(X_train, y_train)
    val_dataset = HSIDataset(X_val, y_val) # Can be empty if ratios lead to it
    test_dataset = HSIDataset(X_test, y_test)

    print(f"Train dataset size: {len(train_dataset)}")
    if len(val_dataset) > 0:
      sample_patch, sample_label = train_dataset[0]
      print(f"Sample patch shape: {sample_patch.shape}, type: {sample_patch.dtype}") # Should be (C_pca, H, W)
      print(f"Sample label: {sample_label}, type: {sample_label.dtype}")

      train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
      val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
      test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

      # Check a batch
      for patches_batch, labels_batch in train_loader:
          print(f"\nBatch patches shape: {patches_batch.shape}") # (B, C_pca, H, W)
          print(f"Batch labels shape: {labels_batch.shape}")
          break
    else:
        print("Validation dataset is empty. Skipping DataLoader test for val set.")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        for patches_batch, labels_batch in train_loader:
            print(f"\nBatch patches shape: {patches_batch.shape}")
            print(f"Batch labels shape: {labels_batch.shape}")
            break