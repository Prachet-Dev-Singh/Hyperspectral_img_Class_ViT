# config.py

# --- Dataset and Preprocessing ---
DATASET_NAME = "IndianPines"
# Ensure these .mat files are in this path or will be downloaded here
DATA_PATH = "./"
INDIAN_PINES_URL_DATA = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
INDIAN_PINES_URL_GT = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
NUM_PCA_COMPONENTS = 30
PATCH_SIZE = 9  # Spatial size of the HSI patch (e.g., 9x9)
TEST_RATIO = 0.8  # 20% for training, 80% for testing
VAL_RATIO_FROM_TRAIN = 0.1 # 10% of the initial training set for validation
RANDOM_SEED = 42 # Crucial for reproducible splits

# --- ViT Model Parameters ---
# The input to ViT is an HSI patch of size (PATCH_SIZE, PATCH_SIZE) with NUM_PCA_COMPONENTS channels
VIT_IMG_SIZE = (PATCH_SIZE, PATCH_SIZE) # Spatial dimensions of the input HSI patch
VIT_SUB_PATCH_SIZE = 3 # ViT will divide the VIT_IMG_SIZE into sub-patches of this size
                       # e.g., if PATCH_SIZE is 9 and VIT_SUB_PATCH_SIZE is 3, we get (9/3)*(9/3) = 9 ViT tokens
EMBED_DIM = 128
NUM_HEADS = 8
NUM_TRANSFORMER_BLOCKS = 6 # N: Total number of transformer blocks (and thus classifiers)
MLP_TRANSFORMER_HIDDEN_LAYERS = [256, 256, 256, 256] # For the 5-layer MLP in each transformer block
DROPOUT_RATE = 0.1

# --- Training Parameters ---
BATCH_SIZE = 64
EPOCHS = 100 # Start with a smaller number for quick tests, e.g., 50-100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# --- Loss Weights ---
# Loss Source 1: Cross-Entropy
LAMBDA_CE_FINAL = 1.0       # Weight for CE loss of the final (deepest) classifier
LAMBDA_CE_AUX = 0.5         # Weight for the sum of CE losses of shallow classifiers

# Loss Source 2: KL Divergence (Deepest classifier as teacher)
LAMBDA_KL_DISTILL = 0.7     # Weight for KL divergence loss (deepest teaches shallow logits)
DISTILLATION_TEMP = 2.0     # Temperature for softmax in KL divergence

# Loss Source 3: L2 Hint Loss (Deepest classifier's features as teacher)
LAMBDA_L2_HINT = 0.001        # Weight for L2 loss between feature maps

# --- Dynamically set by the script ---
NUM_CLASSES_ACTUAL = None # Will be determined after loading GT