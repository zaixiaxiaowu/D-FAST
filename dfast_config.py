# -*- coding: utf-8 -*-
"""
Configuration for D-FAST (Dynamic Fusion of Atomic and Semantic Towers).

Hyperparameters follow the experimental setup described in the paper.
Industrial-specific paths, multi-task heads, and unrelated modules are removed.
"""

# ---------------------------------------------------------------------------
# TGSQ (Taxonomy-Guided Semantic Quantization)
# ---------------------------------------------------------------------------
TGSQ_CONFIG = {
    # Industry taxonomy depth (M_tax) and residual-quantization depth (M_sem).
    "num_taxonomy_levels": 3,
    "num_rq_layers": 3,
    "codebook_size": 256,
    # Dual-view embedding dimensions (paper Section 4.1).
    "dense_sid_dim": 256,
    "sparse_sid_dim": 16,
    # Total hierarchical depth M = M_tax + M_sem.
    "total_semantic_depth": 6,
    # Stage-1 TGSQ pretraining.
    "tgsq_pretrain_epochs": 10,
    "tgsq_learning_rate": 0.001,
    "tgsq_batch_size": 4096,
}

# ---------------------------------------------------------------------------
# DDTA (Decoupled Dual-Tower Architecture)
# ---------------------------------------------------------------------------
TOWER_CONFIG = {
    # Shared MLP hidden sizes for SPT and AMT (excluding the output logit).
    "hidden_sizes": [1024, 256, 128],
    # Base embedding dimension for atomic sparse hash features (d_a).
    "atomic_emb_dim": 16,
    # User/context embedding dimension after lookup (flattened).
    "user_context_dim": None,  # Resolved at runtime from feature config.
    "dropout_rate": 0.0,
    "use_layer_norm": True,
}

# ---------------------------------------------------------------------------
# MCLF (Monotonicity-Constrained Lifecycle Fusion)
# ---------------------------------------------------------------------------
MCLF_CONFIG = {
    # F_1 inflection threshold tau* and adaptive scaling eta (paper Eq. 325-334).
    "tau_star": 10.0,
    "eta": 0.3,
    # Lifecycle stages: New, Cold, Growing, Mature.
    "num_lifecycle_stages": 4,
    "stage_emb_dim": 16,
    # Posterior statistics: [clicks, conversions, cost].
    "num_stat_features": 3,
    # Disable Softplus constraint for the w/o Mono ablation.
    "use_monotonic_constraint": True,
}

# ---------------------------------------------------------------------------
# Optimization objective (paper Eq. 367)
# ---------------------------------------------------------------------------
LOSS_CONFIG = {
    "aux_loss_weight": 0.3,   # lambda
    "l2_reg_weight": 0.05,    # gamma
}

# ---------------------------------------------------------------------------
# Stage-2 joint training
# ---------------------------------------------------------------------------
TRAIN_CONFIG = {
    "joint_epochs": 1,
    "sparse_learning_rate": 0.05,   # AdaGrad for sparse embeddings
    "dense_learning_rate": 0.001,   # AdamW for dense parameters
    "batch_size": 4096,
    "adagrad_eps": 1e-4,
    "adagrad_decay": 0.0,
}

# ---------------------------------------------------------------------------
# Feature groups (abstract names aligned with paper notation)
# ---------------------------------------------------------------------------
# x_a: ad-level atomic hash embeddings (campaign / unit / creative IDs, etc.).
ATOMIC_SPARSE_FEAT_CONFIGS = [
    {"name": "atomic_sparse_main", "num_fields": 410, "emb_dim": 16},
    {"name": "atomic_sparse_tail", "num_fields": 45, "emb_dim": 16},
]

# z_s components materialized offline by TGSQ Stage 0/1.
SEMANTIC_FEAT_CONFIGS = {
    # Dense prior view e_d (dual-view fusion output, dim 256 in experiments).
    "dense_semantic": {"name": "item_dense_semantic_sid", "dim": 256},
    # Sparse hierarchical view e_s: M=6 levels, each with d_spr=16.
    "sparse_semantic_ids": {
        "name": "item_sparse_semantic_ids",
        "num_levels": 6,
        "emb_dim": 16,
    },
}

# x_u, x_c: user profile and request context embeddings.
USER_CONTEXT_FEAT_CONFIGS = [
    {"name": "user_profile_sparse", "num_fields": 410, "emb_dim": 16},
    {"name": "context_sparse", "num_fields": 35, "emb_dim": 16},
]

# Item-level posterior statistics v = [v_clk, v_cvt, v_cst] for MCLF.
ITEM_STAT_FEAT_CONFIGS = [
    {"name": "item_click_count", "dim": 1},
    {"name": "item_conversion_count", "dim": 1},
    {"name": "item_cost_total", "dim": 1},
]

# Mapping from legacy industrial feature names (for reference when adapting data pipelines).
LEGACY_FEATURE_ALIAS = {
    "item_dense_semantic_sid": "ExtractUnitSdpaSemanticIdEmb64D",
    "item_click_count": "ExtractPhotoDenseColdStartClickNum",
    "item_conversion_count": "ExtractPhotoDenseColdStartLpsNum",
    "item_cost_total": "ExtractPhotoDenseColdStartCostTotal",
}
