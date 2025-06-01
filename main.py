# main.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import csv # Import the csv module
from datetime import datetime # For timestamping filenames
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

# Import from our project modules
import config
from utils import set_seeds, ensure_dir
from download_data import download_indian_pines
from data_preprocessing import get_prepared_data # We'll use this
# from data_preprocessing import get_prepared_data_and_processors # If you implement PCA/scaler saving for viz
from hsi_dataset import HSIDataset
from model import VisionTransformerWithAuxHeads
# from losses import compute_total_loss # Used internally by train_one_epoch
from train_eval import train_one_epoch, evaluate

def main_experiment():
    # --- 1. Initial Setup ---
    set_seeds(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure directories for saving models and results
    base_results_dir = "./experiment_results"
    ensure_dir(base_results_dir)
    
    experiment_dir_name = f"{config.DATASET_NAME}_ViTSD_{timestamp}"
    current_experiment_dir = os.path.join(base_results_dir, experiment_dir_name)
    ensure_dir(current_experiment_dir)

    model_save_path = os.path.join(current_experiment_dir, f"best_model.pth")
    epoch_log_path = os.path.join(current_experiment_dir, f"epoch_log_{timestamp}.csv")
    test_report_path = os.path.join(current_experiment_dir, f"test_classification_report_{timestamp}.txt")
    test_metrics_csv_path = os.path.join(current_experiment_dir, f"test_summary_metrics_{timestamp}.csv")


    # --- CSV Setup for Epoch Log ---
    epoch_csv_header = [
        "Epoch", "LR", "Avg_Total_Loss",
        "Avg_CE_Final", "Avg_CE_Aux", "Avg_KL_Distill", "Avg_L2_Hint",
        "Val_Accuracy"
    ]
    with open(epoch_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(epoch_csv_header)

    # --- 2. Data Preparation ---
    print("\n--- Starting Data Preparation ---")
    if not download_indian_pines():
        print("Dataset download/check failed. Exiting.")
        return
    
    # If you were to implement visualization, you'd use:
    # X_train, y_train, X_val, y_val, X_test, y_test, fitted_pca, fitted_scaler, original_gt = get_prepared_data_and_processors()
    X_train, y_train, X_val, y_val, X_test, y_test = get_prepared_data()
    print(f"Number of classes determined: {config.NUM_CLASSES_ACTUAL}")
    if config.NUM_CLASSES_ACTUAL is None or config.NUM_CLASSES_ACTUAL <=0:
        raise ValueError("Number of classes not determined correctly.")

    # --- 3. Create DataLoaders ---
    print("\n--- Creating DataLoaders ---")
    train_dataset = HSIDataset(X_train, y_train)
    val_dataset = None
    if X_val.shape[0] > 0:
        val_dataset = HSIDataset(X_val, y_val)
    else:
        print("Warning: Validation set is empty. No validation will be performed during training.")
    test_dataset = HSIDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Train loader: {len(train_loader)} batches. Val loader: {len(val_loader) if val_loader else 0} batches. Test loader: {len(test_loader)} batches.")

    # --- 4. Initialize Model, Optimizer, Scheduler ---
    print("\n--- Initializing Model and Optimizer ---")
    model = VisionTransformerWithAuxHeads(
        img_size=config.VIT_IMG_SIZE,
        vit_sub_patch_size=config.VIT_SUB_PATCH_SIZE,
        in_channels=config.NUM_PCA_COMPONENTS,
        num_classes=config.NUM_CLASSES_ACTUAL,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_transformer_blocks=config.NUM_TRANSFORMER_BLOCKS,
        mlp_hidden_layers_transformer=config.MLP_TRANSFORMER_HIDDEN_LAYERS,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    
    try:
        from torchinfo import summary
        summary(model, input_size=(config.BATCH_SIZE, config.NUM_PCA_COMPONENTS, config.VIT_IMG_SIZE[0], config.VIT_IMG_SIZE[1]), verbose=0) # verbose=0 for less output
        print("Model summary generated (check full log if not displayed here).")
    except Exception as e:
        print(f"torchinfo summary failed or not installed: {e}")

    # --- 5. Training Loop ---
    print("\n--- Starting Training ---")
    best_val_accuracy = 0.0
    best_epoch = -1

    for epoch in range(1, config.EPOCHS + 1):
        avg_loss, avg_components = train_one_epoch(
            model, train_loader, optimizer, device, epoch, config.EPOCHS
        )
        
        current_lr = scheduler.get_last_lr()[0]
        log_str = (f"Epoch [{epoch}/{config.EPOCHS}] | LR: {current_lr:.2e} | "
                   f"Avg Loss: {avg_loss:.4f} (CE_F: {avg_components.get('ce_final',0):.3f}, "
                   f"CE_A: {avg_components.get('ce_aux_sum',0):.3f}, KL: {avg_components.get('kl_distill_sum',0):.3f}, "
                   f"L2_H: {avg_components.get('l2_hint_sum',0):.4f})") # L2_H with 4 decimal places

        current_val_accuracy = -1.0 # Initialize for CSV logging
        if val_loader:
            val_accuracy, _, _ = evaluate(model, val_loader, device, epoch, config.EPOCHS, eval_type="Validation")
            log_str += f" | Val Acc: {val_accuracy:.2f}%"
            current_val_accuracy = val_accuracy
        else:
            log_str += " | No validation set."
        print(log_str)

        # Log epoch data to CSV
        epoch_data_row = [
            epoch, f"{current_lr:.2e}", f"{avg_loss:.4f}",
            f"{avg_components.get('ce_final',0):.4f}", f"{avg_components.get('ce_aux_sum',0):.4f}",
            f"{avg_components.get('kl_distill_sum',0):.4f}", f"{avg_components.get('l2_hint_sum',0):.4f}",
            f"{current_val_accuracy:.2f}" if current_val_accuracy != -1.0 else "N/A"
        ]
        with open(epoch_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(epoch_data_row)
        
        scheduler.step()

        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            print(f"  New best validation accuracy: {best_val_accuracy:.2f}%. Model saved to {model_save_path}")
        elif not val_loader and epoch == config.EPOCHS:
             torch.save(model.state_dict(), model_save_path)
             print(f"  No validation. Saved model from final epoch to {model_save_path}")

    print("\n--- Training Finished ---")
    if best_epoch != -1:
        print(f"Best Validation Accuracy: {best_val_accuracy:.2f}% at Epoch {best_epoch}")
    else:
        print("Training complete. No validation performed or no improvement seen/model saved.")

    # --- 6. Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set with Best Model ---")
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Loaded best model from {model_save_path}")
    else:
        print(f"Warning: Best model path {model_save_path} not found. Evaluating with the last state of the model.")

    test_accuracy, test_preds, test_gt_labels = evaluate(model, test_loader, device, eval_type="Test")
    print(f"Test Set Accuracy (Overall Accuracy - OA): {test_accuracy:.2f}%")

    # --- Log Test Metrics ---
    oa_val = test_accuracy / 100.0 # as a fraction
    aa_val = 0.0
    kappa_val = 0.0
    
    # Dictionary to store per-class metrics for CSV
    per_class_metrics_for_csv = {} 

    if len(test_gt_labels) > 0 and len(test_preds) > 0 :
        class_names_for_report = [f"Class {i}" for i in range(config.NUM_CLASSES_ACTUAL)]
        try:
            report_text = classification_report(test_gt_labels, test_preds, target_names=class_names_for_report, digits=4, zero_division=0)
            print("\nTest Set Classification Report:")
            print(report_text)
            with open(test_report_path, 'w') as f:
                f.write(f"Test Set Accuracy (Overall Accuracy - OA): {test_accuracy:.2f}%\n\n")
                f.write(report_text)
            
            kappa_val = cohen_kappa_score(test_gt_labels, test_preds)
            print(f"Cohen's Kappa: {kappa_val:.4f}")
            
            cm = confusion_matrix(test_gt_labels, test_preds)
            # Calculate per-class accuracy for AA
            class_accuracies = np.diag(cm) / np.sum(cm, axis=1)
            aa_val = np.mean(class_accuracies[~np.isnan(class_accuracies)]) # Exclude NaN if a class had 0 samples in GT sum
            print(f"Average Accuracy (AA): {aa_val:.4f}")

            with open(test_report_path, 'a') as f: # Append Kappa and AA
                f.write(f"\nCohen's Kappa: {kappa_val:.4f}\n")
                f.write(f"Average Accuracy (AA): {aa_val:.4f}\n")

            # Prepare per-class metrics for CSV
            report_dict = classification_report(test_gt_labels, test_preds, target_names=class_names_for_report, digits=4, zero_division=0, output_dict=True)
            for class_name in class_names_for_report:
                if class_name in report_dict:
                    per_class_metrics_for_csv[f"{class_name}_precision"] = report_dict[class_name]['precision']
                    per_class_metrics_for_csv[f"{class_name}_recall"] = report_dict[class_name]['recall']
                    per_class_metrics_for_csv[f"{class_name}_f1-score"] = report_dict[class_name]['f1-score']
                    per_class_metrics_for_csv[f"{class_name}_support"] = report_dict[class_name]['support']
                else: # Should not happen if target_names are aligned
                    per_class_metrics_for_csv[f"{class_name}_precision"] = 0.0
                    per_class_metrics_for_csv[f"{class_name}_recall"] = 0.0
                    per_class_metrics_for_csv[f"{class_name}_f1-score"] = 0.0
                    per_class_metrics_for_csv[f"{class_name}_support"] = 0

        except Exception as e:
            print(f"Error generating classification report for logging: {e}")
    else:
        print("Not enough data in test set for detailed report.")

    # --- Write Test Summary Metrics to CSV ---
    test_summary_header = ["OA", "AA", "Kappa"] + sorted(list(per_class_metrics_for_csv.keys())) # Add per-class metrics
    test_summary_data = [f"{oa_val:.4f}", f"{aa_val:.4f}", f"{kappa_val:.4f}"] + \
                        [f"{per_class_metrics_for_csv[k]:.4f}" if isinstance(per_class_metrics_for_csv.get(k,0), float) 
                         else per_class_metrics_for_csv.get(k,0) 
                         for k in sorted(list(per_class_metrics_for_csv.keys()))]

    with open(test_metrics_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(test_summary_header)
        writer.writerow(test_summary_data)
    print(f"\nTest summary metrics saved to: {test_metrics_csv_path}")
    print(f"Epoch log saved to: {epoch_log_path}")
    print(f"Full test report saved to: {test_report_path}")
    print(f"Best model saved to: {model_save_path}")


    print("\n--- Experiment Finished ---")


if __name__ == '__main__':
    main_experiment()