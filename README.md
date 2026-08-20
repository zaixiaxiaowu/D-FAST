# D-FAST

**D**ynamic **F**usion of **A**tomic and **S**emantic **T**owers

Reference implementation for conversion rate (CVR) prediction under the **dual cold-start problem** in Single Dynamic Product Ads (SDPA).

> Anonymous release accompanying a WSDM submission. Author and affiliation details will be updated upon acceptance.

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Input Specification](#input-specification)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Ablation Switches](#ablation-switches)
- [Scope and Limitations](#scope-and-limitations)
- [License](#license)

---

## Motivation

In SDPA (Single Dynamic Product Ads), recommendation systems face a **dual cold-start problem**:

| Problem | Description |
|---------|-------------|
| **Natural Cold-Start** | New items lack historical interactions, causing severe sparsity in high-dimensional ID features. |
| **Infrastructure-Driven Fragmentation** | The "One-Item to Multi-Ads" paradigm duplicates the same underlying item across many ad infrastructures (campaigns, units, creatives), inflating the feature space and blocking knowledge reuse. |

D-FAST addresses both issues by:

1. Compressing fragmented ad instances into a **stable hierarchical semantic space** (TGSQ).
2. **Physically decoupling** semantic-prior learning from atomic-ID memorization (DDTA).
3. **Dynamically shifting** prediction reliance from global semantics to individual atomic features as item-level data accumulates (MCLF).

---

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │              TGSQ (frozen in Stage 2)      │
                         │  e_d (dense SID)  +  e_s (sparse SID)   │
                         │              ──►  z_s                     │
                         └───────────────┬─────────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
              ▼                          ▼                          │
   ┌─────────────────────┐    ┌─────────────────────┐              │
   │   SPT (Semantic     │    │   AMT (Atomic       │              │
   │   Prior Tower)      │    │   Memorization      │              │
   │                     │    │   Tower)            │              │
   │  Input: z_s ⊕ x_u   │    │  Input: x_a ⊕       │              │
   │         ⊕ x_c       │    │         sg(z_s) ⊕   │              │
   │  (masks x_a)        │    │         x_u ⊕ x_c   │              │
   │         │           │    │         │           │              │
   │         ▼           │    │         ▼           │              │
   │       y_p (pCVR)    │    │       y_m (pCVR)    │              │
   └──────────┬──────────┘    └──────────┬──────────┘              │
              │                          │                          │
              └────────────┬─────────────┘                          │
                           │                                        │
                           ▼                                        │
              ┌────────────────────────┐                            │
              │  MCLF Gating Network   │◄── h_n = log(v + 1)       │
              │  α = σ(h_n^T W_α + b)  │◄── h_stg = Embed(S(v_cvt))│
              └────────────┬───────────┘                            │
                           │                                        │
                           ▼                                        │
              y_f = α · y_m + (1 − α) · y_p  ──►  final pCVR       │
```

### Module 1: TGSQ — Taxonomy-Guided Semantic Quantization

TGSQ maps multi-modal item semantics into a dual-view representation **z_s = e_d ⊕ e_s**:

| View | Symbol | Description |
|------|--------|-------------|
| Dense Prior | **e_d** | Continuous semantic vector aligned with industry taxonomy (dim = 256). |
| Sparse Hierarchical | **e_s** | M = 6 levels of learned sparse embeddings (3 taxonomy + 3 RQ layers, dim = 16 each). |

**Offline preparation (Stage 0):** LLM-based trait extraction, text encoding, and K-Means RQ codebook generation are performed offline. This repository consumes the **materialized** dense and sparse semantic features rather than re-running LLM pipelines.

**In-code behavior:** `TGSQModule.fuse_dual_view()` concatenates precomputed `dense_semantic` with sparse ID lookups through `E_spr`.

### Module 2: DDTA — Decoupled Dual-Tower Architecture

| Tower | Role | Input | Output |
|-------|------|-------|--------|
| **SPT** | Global semantic prior | z_s ⊕ x_u ⊕ x_c | y_p |
| **AMT** | Ad-level atomic memorization | x_a ⊕ sg(z_s) ⊕ x_u ⊕ x_c | y_m |

Key design choices:

- SPT **masks all atomic infrastructure IDs** (x_a), forcing reliance on semantics during cold-start.
- AMT applies **stop-gradient** on z_s to prevent atomic gradients from corrupting the semantic pathway.
- Both towers use identical MLP backbones: `[1024 → 256 → 128 → 1]` with ReLU and layer normalization.

### Module 3: MCLF — Monotonicity-Constrained Lifecycle Fusion

MCLF computes a gating weight **α ∈ (0, 1)** that controls the fusion:

```
y_f = α · y_m + (1 − α) · y_p
```

**Lifecycle stages** (based on item-level conversion count v_cvt):

| Stage | Condition | Model behavior |
|-------|-----------|----------------|
| New | v_cvt = 0 | Fully rely on SPT semantic prior |
| Cold | 0 < v_cvt ≤ η·τ* | Semantic prior dominates; atomic IDs are noisy |
| Growing | η·τ* < v_cvt ≤ τ* | Transition phase; α increases steadily |
| Mature | v_cvt > τ* | Atomic memorization dominates |

Default hyperparameters: **τ* = 10**, **η = 0.3**.

**Monotonicity constraint:** W_α is parameterized via Softplus to ensure element-wise positive weights, so α increases monotonically with log-transformed statistics h_n = log(v + 1) within each fixed lifecycle stage.

---

## Repository Structure

```
.
├── dfast_config.py      # Hyperparameters and feature group definitions
├── dfast_model.py       # TGSQ, MCLF, SPT/AMT towers, loss, and graph builder
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── README.md            # This file
```

---

## Installation

**Requirements:** Python 3.7+, TensorFlow 1.x compatible API (`tf.compat.v1`).

```bash
git clone https://github.com/zaixiaxiaowu/D-FAST.git
cd D-FAST
pip install -r requirements.txt
```

Verify installation:

```python
import tensorflow as tf
import dfast_model
import dfast_config
print("TensorFlow:", tf.__version__)
print("D-FAST config loaded:", dfast_config.TGSQ_CONFIG["codebook_size"])
```

---

## Input Specification

The model expects five tensor groups per batch. Dimensions below match the paper's experimental setup.

| Placeholder | Shape | Description | Paper notation |
|-------------|-------|-------------|----------------|
| `atomic_emb` | `[B, 7280]` | Flattened hash embeddings of ad-level atomic sparse features (455 fields × 16 dim) | x_a |
| `dense_semantic` | `[B, 256]` | TGSQ dense prior view (materialized offline) | e_d |
| `sparse_semantic_ids` | `[B, 6]` | Hierarchical semantic ID indices (int32) | I |
| `user_context_emb` | `[B, 7120]` | Flattened user + context sparse embeddings (445 fields × 16 dim) | x_u ⊕ x_c |
| `item_stats` | `[B, 3]` | Item-level posterior stats: `[clicks, conversions, cost]` | v |
| `labels` | `[B]` | Binary conversion labels {0, 1} | y |

> **Note:** Atomic and user/context embedding dimensions are derived from `dfast_config.py`. You must pre-compute hash embedding lookups in your data pipeline before feeding tensors into this model.

---

## Quick Start

### Build the training graph

```python
import numpy as np
import tensorflow as tf
from dfast_model import build_training_graph

tf.compat.v1.disable_eager_execution()

placeholders, outputs, loss_dict = build_training_graph(batch_size=32)

train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(
    loss_dict["loss_total"]
)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    feed = {
        placeholders["atomic_emb"]: np.random.randn(32, 7280).astype(np.float32),
        placeholders["dense_semantic"]: np.random.randn(32, 256).astype(np.float32),
        placeholders["sparse_semantic_ids"]: np.random.randint(0, 256, size=(32, 6)).astype(np.int32),
        placeholders["user_context_emb"]: np.random.randn(32, 7120).astype(np.float32),
        placeholders["item_stats"]: np.abs(np.random.randn(32, 3)).astype(np.float32),
        placeholders["labels"]: np.random.randint(0, 2, size=(32,)).astype(np.float32),
    }

    result = sess.run(
        {
            "y_f": outputs["y_f"],
            "y_p": outputs["y_p"],
            "y_m": outputs["y_m"],
            "alpha": outputs["alpha"],
            "stage_ids": outputs["stage_ids"],
            "loss": loss_dict["loss_total"],
        },
        feed_dict=feed,
    )

    print("Fused pCVR (mean):", result["y_f"].mean())
    print("Gating alpha (mean):", result["alpha"].mean())
    print("Lifecycle stages:", result["stage_ids"][:10])
    print("Total loss:", result["loss"])
```

### Key output tensors

| Key | Description |
|-----|-------------|
| `outputs["y_f"]` | Fused pCVR prediction (used for ranking) |
| `outputs["y_p"]` | SPT (semantic prior) branch prediction |
| `outputs["y_m"]` | AMT (atomic memorization) branch prediction |
| `outputs["alpha"]` | Lifecycle gating weight (higher → more AMT reliance) |
| `outputs["stage_ids"]` | Lifecycle stage index: 0=New, 1=Cold, 2=Growing, 3=Mature |
| `outputs["h_n"]` | Log-transformed item statistics |
| `loss_dict["loss_total"]` | Full objective L_tot |
| `loss_dict["loss_f"]` | Fused branch BCE loss |
| `loss_dict["loss_p"]` | SPT auxiliary BCE loss |
| `loss_dict["loss_m"]` | AMT auxiliary BCE loss |

---

## Training Pipeline

D-FAST uses a three-stage training procedure:

```
Stage 0 ──► Stage 1 ──► Stage 2
(offline)   (TGSQ)      (joint D-FAST)
```

| Stage | What runs | Trainable | Frozen |
|-------|-----------|-----------|--------|
| **0 — Offline Semantic Prep** | LLM trait extraction, text encoding, K-Means RQ codebooks | — | LLM encoders, codebooks |
| **1 — TGSQ Pretraining** (10 epochs) | Train E_spr and W_pro | Sparse SID embeddings, projection matrix | LLM encoders, codebooks |
| **2 — Joint D-FAST** (1 epoch) | Train SPT, AMT, MCLF, ranking embeddings | SPT, AMT, MCLF, E_a, E_u, E_c | All TGSQ parameters |

### Stage 1: TGSQ pretraining

```python
from dfast_model import DFASTModel

model = DFASTModel(tgsq_trainable=True, is_training=True)
outputs, loss_dict = model.build_graph(
    atomic_emb=...,
    dense_semantic=...,
    sparse_semantic_ids=...,
    user_context_emb=...,
    item_stats=...,
    labels=...,
)
# Optimize with AdamW, lr=0.001, batch_size=4096
```

### Stage 2: Joint training (recommended setting)

```python
model = DFASTModel(tgsq_trainable=False, is_training=True)
outputs, loss_dict = model.build_graph(...)

# Recommended optimizers (paper Section 4.1):
#   - AdaGrad (lr=0.05) for sparse embedding parameters
#   - AdamW  (lr=0.001) for dense (MLP, MCLF) parameters
```

### Optimization objective

The total loss implemented in `compute_loss()`:

```
L_tot = L_f + λ · [(1 − sg(α)) · L_p + L_m] + (γ/2) · ||Θ||²
```

| Symbol | Default | Meaning |
|--------|---------|---------|
| λ | 0.3 | Auxiliary loss weight |
| γ | 0.05 | L2 regularization coefficient |
| sg(α) | — | Stop-gradient on α for SPT loss weighting only |

As α → 1 (mature items), the SPT auxiliary gradient weight `(1 − sg(α))` → 0, preventing semantic overfitting on items with sufficient atomic data.

---

## Configuration Reference

All hyperparameters live in `dfast_config.py`:

### TGSQ (`TGSQ_CONFIG`)

| Key | Default | Description |
|-----|---------|-------------|
| `num_taxonomy_levels` | 3 | Industry taxonomy depth M_tax |
| `num_rq_layers` | 3 | Residual quantization depth M_sem |
| `codebook_size` | 256 | RQ codebook size K |
| `dense_sid_dim` | 256 | Dense SID dimension d_den |
| `sparse_sid_dim` | 16 | Sparse SID dimension d_spr |
| `total_semantic_depth` | 6 | M = M_tax + M_sem |

### Towers (`TOWER_CONFIG`)

| Key | Default | Description |
|-----|---------|-------------|
| `hidden_sizes` | [1024, 256, 128] | MLP hidden layer sizes |
| `atomic_emb_dim` | 16 | Per-field atomic embedding dim |
| `use_layer_norm` | True | Layer normalization in MLP |

### MCLF (`MCLF_CONFIG`)

| Key | Default | Description |
|-----|---------|-------------|
| `tau_star` | 10.0 | F_1 inflection threshold τ* |
| `eta` | 0.3 | Lifecycle scaling coefficient η |
| `stage_emb_dim` | 16 | Lifecycle stage embedding dim |
| `use_monotonic_constraint` | True | Enable Softplus monotonic gating |

### Loss and Training (`LOSS_CONFIG`, `TRAIN_CONFIG`)

| Key | Default | Description |
|-----|---------|-------------|
| `aux_loss_weight` | 0.3 | λ |
| `l2_reg_weight` | 0.05 | γ |
| `sparse_learning_rate` | 0.05 | AdaGrad lr for sparse params |
| `dense_learning_rate` | 0.001 | AdamW lr for dense params |
| `batch_size` | 4096 | Training batch size |

---

## API Reference

### Classes

#### `TGSQModule`

```python
tgsq = TGSQModule(config=None)
z_s = tgsq.fuse_dual_view(dense_semantic, sparse_semantic_ids, trainable=False)
```

#### `MCLFModule`

```python
mclf = MCLFModule(config=None)
alpha, h_n, stage_ids = mclf.compute_alpha(item_stats, v_cvt)
```

#### `DFASTModel`

```python
model = DFASTModel(
    tgsq_trainable=False,       # False for Stage 2
    is_training=True,
)
outputs = model.forward(atomic_emb, dense_semantic, sparse_semantic_ids,
                        user_context_emb, item_stats)
loss_dict = model.compute_loss(outputs, labels)
```

### Helper functions

| Function | Description |
|----------|-------------|
| `build_placeholders(batch_size=None)` | Create TF placeholders for all inputs |
| `build_training_graph(batch_size, tgsq_trainable, is_training)` | End-to-end graph builder |
| `build_mlp(...)` | Reusable ReLU MLP with optional layer norm |
| `binary_cross_entropy(y_true, y_pred)` | Branch-level BCE loss |

---

## Ablation Switches

Reproduce paper ablations by modifying config or constructor flags:

| Ablation | How to enable |
|----------|---------------|
| **w/o Mono** (no monotonic constraint) | Set `MCLF_CONFIG["use_monotonic_constraint"] = False` |
| **Static Fusion** (fixed α = 0.5) | Replace `alpha` with `0.5` in `forward()` |
| **Hard Switch** | Use `y_p` if v_cvt ≤ τ*, else `y_m` (no soft interpolation) |
| **SID-Single** (no DDTA) | Merge SPT and AMT into a single tower |
| **Base-ID** (no semantics) | Remove z_s from both tower inputs |

---

## Scope and Limitations

This repository provides the **core D-FAST model graph** in TensorFlow. The following components are **not included** and must be implemented in your own data/training infrastructure:

| Component | Status |
|-----------|--------|
| LLM-based multi-modal feature extraction (Stage 0) | Not included — use pre-materialized features |
| K-Means RQ codebook generation | Not included — provide `sparse_semantic_ids` offline |
| Hash embedding lookup for x_a, x_u, x_c | Not included — feed pre-looked-up tensors |
| Distributed training / parameter servers | Not included |
| Full industrial dataset | Not included — proprietary data |

The implementation uses `tf.compat.v1` for graph-mode TensorFlow. For TensorFlow 2.x eager execution, additional refactoring is required.

---

## License

This project is released under the [MIT License](LICENSE).
