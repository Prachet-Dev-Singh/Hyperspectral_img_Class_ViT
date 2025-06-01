# train_eval.py
import torch
import numpy as np
from tqdm import tqdm 

from losses import compute_total_loss

def train_one_epoch(model, dataloader, optimizer, device, current_epoch, total_epochs):
    model.train()
    total_loss_epoch = 0.0
    
    epoch_loss_components = {
        "ce_final": 0.0, "ce_aux_sum": 0.0,
        "kl_distill_sum": 0.0, "l2_hint_sum": 0.0
    }

    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch}/{total_epochs} [Training]", leave=False)
    for patches, labels in progress_bar:
        patches, labels = patches.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        model_outputs = model(patches) 
        
        loss, batch_loss_components = compute_total_loss(model_outputs, labels)
        
        if torch.isnan(loss) or torch.isinf(loss): # Added check for inf loss
            print(f"NaN or Inf loss detected at epoch {current_epoch}, batch. Loss: {loss.item()}. Components: {batch_loss_components}")
            # If this happens, consider stopping or reducing LR further.
            # For now, skip update for this batch.
            continue 

        loss.backward()
        
        # MODIFICATION: Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradients
        
        optimizer.step()
        
        total_loss_epoch += loss.item()
        for key in epoch_loss_components:
            if key in batch_loss_components:
                 epoch_loss_components[key] += batch_loss_components[key]

        progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}")
            
    num_batches = len(dataloader)
    if num_batches == 0: return 0.0, {}

    avg_epoch_loss = total_loss_epoch / num_batches
    avg_epoch_components = {k: v / num_batches for k, v in epoch_loss_components.items()}
    
    return avg_epoch_loss, avg_epoch_components

def evaluate(model, dataloader, device, current_epoch=None, total_epochs=None, eval_type="Validation"):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    desc_text = f"Epoch {current_epoch}/{total_epochs} [{eval_type}]" if current_epoch else f"[{eval_type}]"
    progress_bar = tqdm(dataloader, desc=desc_text, leave=False)

    with torch.no_grad():
        for patches, labels_batch in progress_bar: # renamed labels to labels_batch to avoid conflict
            patches, labels_batch = patches.to(device), labels_batch.to(device) # ensure labels_batch is on device
            
            model_outputs = model(patches)
            final_logits = model_outputs[0][-1] 
            
            _, predicted = torch.max(final_logits.data, 1)
            
            total_samples += labels_batch.size(0)
            total_correct += (predicted == labels_batch).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy()) # Use labels_batch here
            
    if total_samples == 0: return 0.0, [], []
    
    accuracy = 100.0 * total_correct / total_samples
    return accuracy, np.array(all_preds), np.array(all_labels)


if __name__ == '__main__':
    print("--- Testing train_eval.py functions (basic structural test) ---")

    import config as cfg
    from utils import set_seeds
    from model import VisionTransformerWithAuxHeads
    from hsi_dataset import HSIDataset 
    from torch.utils.data import DataLoader
    import torch.optim as optim

    set_seeds(cfg.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if cfg.NUM_CLASSES_ACTUAL is None:
        cfg.NUM_CLASSES_ACTUAL = 3 
        print(f"Set cfg.NUM_CLASSES_ACTUAL to {cfg.NUM_CLASSES_ACTUAL} for test.")
    
    test_model = VisionTransformerWithAuxHeads(
        img_size=cfg.VIT_IMG_SIZE, vit_sub_patch_size=cfg.VIT_SUB_PATCH_SIZE,
        in_channels=cfg.NUM_PCA_COMPONENTS, num_classes=cfg.NUM_CLASSES_ACTUAL,
        embed_dim=cfg.EMBED_DIM, num_heads=cfg.NUM_HEADS,
        num_transformer_blocks=2, 
        mlp_hidden_layers_transformer=cfg.MLP_TRANSFORMER_HIDDEN_LAYERS[:1]*2 if cfg.MLP_TRANSFORMER_HIDDEN_LAYERS and len(cfg.MLP_TRANSFORMER_HIDDEN_LAYERS)>0 else [64,64], 
        dropout_rate=cfg.DROPOUT_RATE
    ).to(device)

    dummy_patches = np.random.randn(cfg.BATCH_SIZE * 2, cfg.PATCH_SIZE, cfg.PATCH_SIZE, cfg.NUM_PCA_COMPONENTS).astype(np.float32)
    dummy_labels = np.random.randint(0, cfg.NUM_CLASSES_ACTUAL, cfg.BATCH_SIZE * 2)
    
    dummy_dataset = HSIDataset(dummy_patches, dummy_labels)
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=cfg.BATCH_SIZE)

    optimizer = optim.AdamW(test_model.parameters(), lr=1e-4)

    print("\nTesting train_one_epoch...")
    avg_loss, avg_components = train_one_epoch(test_model, dummy_dataloader, optimizer, device, 1, 1)
    print(f"  Avg Epoch Loss: {avg_loss:.4f}")
    print(f"  Avg Loss Components: {avg_components}")

    print("\nTesting evaluate...")
    accuracy, _, _ = evaluate(test_model, dummy_dataloader, device, eval_type="Test Evaluation")
    print(f"  Evaluation Accuracy: {accuracy:.2f}%")

    print("\ntrain_eval.py basic test finished.")