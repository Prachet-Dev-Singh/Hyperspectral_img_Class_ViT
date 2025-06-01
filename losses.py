# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F # Make sure F is imported

from config import (
    LAMBDA_CE_FINAL, LAMBDA_CE_AUX,
    LAMBDA_KL_DISTILL, DISTILLATION_TEMP,
    LAMBDA_L2_HINT
)

# Standard Cross-Entropy Loss
criterion_ce = nn.CrossEntropyLoss()

# KL Divergence Loss for distillation
criterion_kl_div = nn.KLDivLoss(reduction='batchmean', log_target=False)

# L2 Loss (Mean Squared Error) for hint loss
criterion_l2 = nn.MSELoss()


def compute_total_loss(model_outputs, labels):
    """
    Computes the total loss based on the three specified sources.

    Args:
        model_outputs (tuple): A tuple containing:
            - aux_logits_list (list of Tensors): Logits from each classifier (shallow to deep).
            - intermediate_cls_features_list (list of Tensors): CLS token features from each block's
                                                              "bottleneck" (before MLP).
        labels (Tensor): Ground truth labels.

    Returns:
        total_loss (Tensor): The final combined loss.
        loss_components (dict): A dictionary containing the individual loss values.
    """
    aux_logits_list, intermediate_cls_features_list = model_outputs
    num_classifiers = len(aux_logits_list)

    if num_classifiers == 0:
        return torch.tensor(0.0, device=labels.device, requires_grad=True), {}
    
    # --- Loss Source 1: Cross-Entropy Losses ---
    final_logits = aux_logits_list[-1]
    loss_ce_final = criterion_ce(final_logits, labels)

    loss_ce_aux_sum = torch.tensor(0.0, device=labels.device)
    if num_classifiers > 1:
        for i in range(num_classifiers - 1):
            shallow_logits = aux_logits_list[i]
            loss_ce_aux_sum += criterion_ce(shallow_logits, labels)
    
    # --- Loss Source 2: KL Divergence Loss (Deepest as Teacher) ---
    loss_kl_distill_sum = torch.tensor(0.0, device=labels.device)
    if num_classifiers > 1:
        with torch.no_grad():
            probs_teacher = F.softmax(final_logits / DISTILLATION_TEMP, dim=1)

        for i in range(num_classifiers - 1):
            student_logits = aux_logits_list[i]
            log_probs_student = F.log_softmax(student_logits / DISTILLATION_TEMP, dim=1)
            kl_loss = criterion_kl_div(log_probs_student, probs_teacher) * (DISTILLATION_TEMP ** 2)
            loss_kl_distill_sum += kl_loss

    # --- Loss Source 3: L2 Hint Loss (Features from Deepest as Teacher) ---
    loss_l2_hint_sum = torch.tensor(0.0, device=labels.device)
    if num_classifiers > 1 and len(intermediate_cls_features_list) == num_classifiers:
        # Teacher features: From the deepest classifier's bottleneck layer (detached)
        teacher_features = intermediate_cls_features_list[-1].detach()

        for i in range(num_classifiers - 1): # Iterate over shallow classifiers' features
            student_features = intermediate_cls_features_list[i]
            
            # MODIFICATION: Normalize features before L2 comparison
            student_features_norm = F.normalize(student_features, p=2, dim=1)
            teacher_features_norm = F.normalize(teacher_features, p=2, dim=1) # teacher_features is already detached
            
            l2_loss = criterion_l2(student_features_norm, teacher_features_norm)
            loss_l2_hint_sum += l2_loss
            
    elif num_classifiers > 1 and len(intermediate_cls_features_list) != num_classifiers:
        print("Warning: Mismatch between number of classifiers and feature lists for L2 hint loss. Skipping L2 loss.")


    # --- Combine Losses ---
    total_loss = (LAMBDA_CE_FINAL * loss_ce_final +
                  LAMBDA_CE_AUX * loss_ce_aux_sum +
                  LAMBDA_KL_DISTILL * loss_kl_distill_sum +
                  LAMBDA_L2_HINT * loss_l2_hint_sum) # LAMBDA_L2_HINT is now smaller

    loss_components = {
        "total_loss": total_loss.item(),
        "ce_final": loss_ce_final.item(),
        "ce_aux_sum": loss_ce_aux_sum.item() if isinstance(loss_ce_aux_sum, torch.Tensor) else loss_ce_aux_sum,
        "kl_distill_sum": loss_kl_distill_sum.item() if isinstance(loss_kl_distill_sum, torch.Tensor) else loss_kl_distill_sum,
        "l2_hint_sum": loss_l2_hint_sum.item() if isinstance(loss_l2_hint_sum, torch.Tensor) else loss_l2_hint_sum,
    }
    
    return total_loss, loss_components


if __name__ == "__main__":
    # --- Test the loss computation ---
    from utils import set_seeds
    from model import VisionTransformerWithAuxHeads 
    import config as cfg 

    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_batch_size = 4
    if cfg.NUM_CLASSES_ACTUAL is None: 
        print("Setting cfg.NUM_CLASSES_ACTUAL to 5 for loss test.")
        cfg.NUM_CLASSES_ACTUAL = 5
    
    test_num_classes = cfg.NUM_CLASSES_ACTUAL
    test_embed_dim = cfg.EMBED_DIM 
    test_num_blocks = cfg.NUM_TRANSFORMER_BLOCKS 


    print(f"--- Testing Loss Functions with {test_num_blocks} blocks/classifiers ---")
    print(f"Lambdas: CE_Final={LAMBDA_CE_FINAL}, CE_Aux={LAMBDA_CE_AUX}, KL={LAMBDA_KL_DISTILL}, L2_Hint={LAMBDA_L2_HINT}") # Will show new LAMBDA_L2_HINT

    test_model = VisionTransformerWithAuxHeads(
        img_size=cfg.VIT_IMG_SIZE,
        vit_sub_patch_size=cfg.VIT_SUB_PATCH_SIZE,
        in_channels=cfg.NUM_PCA_COMPONENTS,
        num_classes=test_num_classes,
        embed_dim=test_embed_dim,
        num_heads=cfg.NUM_HEADS,
        num_transformer_blocks=test_num_blocks,
        mlp_hidden_layers_transformer=cfg.MLP_TRANSFORMER_HIDDEN_LAYERS,
        dropout_rate=cfg.DROPOUT_RATE
    ).to(device)
    test_model.eval() 

    dummy_hsi_patch_input = torch.randn(
        test_batch_size, cfg.NUM_PCA_COMPONENTS, cfg.VIT_IMG_SIZE[0], cfg.VIT_IMG_SIZE[1]
    ).to(device)
    
    with torch.no_grad(): 
        mock_logits_list, mock_features_list = test_model(dummy_hsi_patch_input)

    mock_labels = torch.randint(0, test_num_classes, (test_batch_size,)).to(device)

    print(f"\nShapes of mock outputs:")
    for i in range(len(mock_logits_list)):
        print(f"  Logits {i}: {mock_logits_list[i].shape}, Features {i}: {mock_features_list[i].shape}")
    print(f"Labels shape: {mock_labels.shape}")

    total_loss_val, components = compute_total_loss(
        (mock_logits_list, mock_features_list), mock_labels
    )

    print(f"\nComputed total loss: {total_loss_val.item():.4f}")
    print("Loss components:")
    for name, value in components.items():
        if isinstance(value, torch.Tensor):
             print(f"  {name}: {value.item():.4f}")
        else:
             print(f"  {name}: {value:.4f}")

    if test_num_blocks == 1:
        print("\n--- Testing with 1 block (no shallow classifiers) ---")
        # This part of test might not run if default num_blocks > 1
        # For it to run, you'd need to ensure test_num_blocks is set to 1 for this specific test path.
        # The re-test by configuring 1 block directly is more reliable for this edge case.
        pass # Placeholder, as the below re-test is better

    print("\n--- Re-testing by configuring 1 block directly ---")
    original_num_blocks = cfg.NUM_TRANSFORMER_BLOCKS
    cfg.NUM_TRANSFORMER_BLOCKS = 1 # Temporarily set to 1 for this test
    test_model_1_block = VisionTransformerWithAuxHeads(
        img_size=cfg.VIT_IMG_SIZE, vit_sub_patch_size=cfg.VIT_SUB_PATCH_SIZE,
        in_channels=cfg.NUM_PCA_COMPONENTS, num_classes=test_num_classes,
        embed_dim=test_embed_dim, num_heads=cfg.NUM_HEADS,
        num_transformer_blocks=1, 
        mlp_hidden_layers_transformer=cfg.MLP_TRANSFORMER_HIDDEN_LAYERS,
        dropout_rate=cfg.DROPOUT_RATE
    ).to(device)
    test_model_1_block.eval()
    with torch.no_grad():
        mock_logits_list_1block, mock_features_list_1block = test_model_1_block(dummy_hsi_patch_input)
    
    total_loss_1block_config, components_1block_config = compute_total_loss(
        (mock_logits_list_1block, mock_features_list_1block), mock_labels
    )
    print(f"Computed total loss (1 block configured): {total_loss_1block_config.item():.4f}")
    print("Loss components (1 block configured):")
    for name, value in components_1block_config.items():
        print(f"  {name}: {value:.4f}")
    cfg.NUM_TRANSFORMER_BLOCKS = original_num_blocks # Reset