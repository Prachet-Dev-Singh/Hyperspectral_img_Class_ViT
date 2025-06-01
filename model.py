# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import specific, static config values
from config import (
    VIT_IMG_SIZE, VIT_SUB_PATCH_SIZE, NUM_PCA_COMPONENTS,
    EMBED_DIM, NUM_HEADS, NUM_TRANSFORMER_BLOCKS,
    MLP_TRANSFORMER_HIDDEN_LAYERS, DROPOUT_RATE
)
# Import the whole config module to access potentially dynamic values like NUM_CLASSES_ACTUAL
import config

class AuxHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes cannot be None for AuxHead initialization.")
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, embed_dim) - typically the CLS token
        return self.fc(x)

class MLP(nn.Module):
    """5-layer MLP for Transformer Block as per blueprint"""
    def __init__(self, in_features, hidden_layers_dims, out_features, dropout_rate):
        super().__init__()
        layers = []
        current_dim = in_features
        # Input -> H1 -> H2 -> H3 -> H4 -> Output
        # This means 4 hidden layers and 5 linear layers in total.
        for h_dim in hidden_layers_dims: # Should have 4 elements for 5 layers
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU()) # Common activation in transformers
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, out_features))
        layers.append(nn.Dropout(dropout_rate)) # Dropout after final linear layer
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_layers, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_hidden_layers, embed_dim, dropout_rate)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        
        # Pre-norm for attention
        normed_x = self.norm1(x)
        attn_output, _ = self.attn(normed_x, normed_x, normed_x)
        features_after_attn_res = x + self.dropout_attn(attn_output)
        
        # Pre-norm for MLP
        normed_features = self.norm2(features_after_attn_res)
        mlp_output = self.mlp(normed_features)
        out = features_after_attn_res + mlp_output

        return out, features_after_attn_res # Return final output and intermediate features

class VisionTransformerWithAuxHeads(nn.Module):
    def __init__(self, img_size, vit_sub_patch_size, in_channels, num_classes,
                 embed_dim, num_heads, num_transformer_blocks,
                 mlp_hidden_layers_transformer, dropout_rate):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_transformer_blocks = num_transformer_blocks
        
        if num_classes is None:
            raise ValueError("num_classes cannot be None for VisionTransformerWithAuxHeads initialization.")
        self.num_classes = num_classes


        assert img_size[0] % vit_sub_patch_size == 0 and img_size[1] % vit_sub_patch_size == 0, \
            f"VIT_IMG_SIZE {img_size} must be divisible by VIT_SUB_PATCH_SIZE {vit_sub_patch_size}."

        self.num_vit_patches_h = img_size[0] // vit_sub_patch_size
        self.num_vit_patches_w = img_size[1] // vit_sub_patch_size
        self.num_vit_patches_total = self.num_vit_patches_h * self.num_vit_patches_w

        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=vit_sub_patch_size,
                                     stride=vit_sub_patch_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_vit_patches_total + 1, embed_dim))
        self.dropout_pos = nn.Dropout(dropout_rate)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_hidden_layers_transformer, dropout_rate)
            for _ in range(num_transformer_blocks)
        ])

        self.aux_heads = nn.ModuleList([
            AuxHead(embed_dim, self.num_classes) # Use self.num_classes
            for _ in range(num_transformer_blocks)
        ])

    def forward(self, x):
        x_embedded = self.patch_embed(x)
        x_embedded = x_embedded.flatten(2)
        x_embedded = x_embedded.transpose(1, 2)

        batch_size = x_embedded.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, x_embedded), dim=1)

        x_seq = x_seq + self.pos_embed
        x_seq = self.dropout_pos(x_seq)

        intermediate_cls_features = []
        aux_logits_list = []

        current_input = x_seq
        for i in range(self.num_transformer_blocks):
            block_output, features_before_mlp = self.transformer_blocks[i](current_input)
            
            cls_token_output_for_clf = block_output[:, 0]
            aux_logits = self.aux_heads[i](cls_token_output_for_clf)
            aux_logits_list.append(aux_logits)

            cls_token_feature_for_hint = features_before_mlp[:, 0]
            intermediate_cls_features.append(cls_token_feature_for_hint)
            
            current_input = block_output

        return aux_logits_list, intermediate_cls_features

if __name__ == "__main__":
    from config import BATCH_SIZE # BATCH_SIZE is static, so direct import is fine
    from utils import set_seeds
    
    set_seeds(42)

    # Ensure config.NUM_CLASSES_ACTUAL is set for the test
    if config.NUM_CLASSES_ACTUAL is None:
        print("Warning: config.NUM_CLASSES_ACTUAL not set. Setting to 16 for test.")
        config.NUM_CLASSES_ACTUAL = 16 # Modify the attribute in the imported config module


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if config.NUM_CLASSES_ACTUAL is valid before model instantiation
    if config.NUM_CLASSES_ACTUAL is None or not isinstance(config.NUM_CLASSES_ACTUAL, int) or config.NUM_CLASSES_ACTUAL <= 0:
        raise ValueError(f"config.NUM_CLASSES_ACTUAL is invalid ({config.NUM_CLASSES_ACTUAL}). It must be a positive integer.")

    model = VisionTransformerWithAuxHeads(
        img_size=VIT_IMG_SIZE,
        vit_sub_patch_size=VIT_SUB_PATCH_SIZE,
        in_channels=NUM_PCA_COMPONENTS,
        num_classes=config.NUM_CLASSES_ACTUAL, # Access the (potentially modified) value from the config module
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        mlp_hidden_layers_transformer=MLP_TRANSFORMER_HIDDEN_LAYERS,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    dummy_input = torch.randn(BATCH_SIZE, NUM_PCA_COMPONENTS, VIT_IMG_SIZE[0], VIT_IMG_SIZE[1]).to(device)

    print(f"\nInput shape: {dummy_input.shape}")
    logits_list, features_list = model(dummy_input)

    print("\n--- Model Output Shapes ---")
    print(f"Number of transformer blocks: {NUM_TRANSFORMER_BLOCKS}")
    print(f"Length of logits_list: {len(logits_list)}")
    print(f"Length of features_list: {len(features_list)}")

    for i in range(len(logits_list)):
        print(f"  Logits from AuxHead {i+1} shape: {logits_list[i].shape}")
        print(f"  Features from Block {i+1} (for hint) shape: {features_list[i].shape}")

    try:
        from torchinfo import summary
        summary(model, input_size=(BATCH_SIZE, NUM_PCA_COMPONENTS, VIT_IMG_SIZE[0], VIT_IMG_SIZE[1]))
    except ImportError:
        print("\ntorchinfo not found. Skipping model summary.")
        print(model)
    except Exception as e:
        print(f"\nError during torchinfo.summary: {e}")
        print(model)