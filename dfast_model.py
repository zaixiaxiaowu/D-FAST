# -*- coding: utf-8 -*-
"""
D-FAST reference implementation extracted from the industrial ranking model.

This file implements the three core modules described in the paper:
  1. TGSQ  - Taxonomy-Guided Semantic Quantization
  2. DDTA  - Decoupled Dual-Tower Architecture (SPT + AMT)
  3. MCLF  - Monotonicity-Constrained Lifecycle Fusion

Dependencies: TensorFlow 1.x compatible API (tf.compat.v1).
Industrial training frameworks (Kai/Klearn), RankMixer, DIN, and multi-task
heads are intentionally removed for open-source release.
"""

from __future__ import absolute_import, division, print_function

import logging
import math

import tensorflow as tf

import dfast_config as cfg

LOG_FORMAT = (
    "%(asctime)s - %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s] - %(message)s"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

tf.compat.v1.disable_eager_execution()


def stop_gradient(x):
    """Stop-gradient operator sg(.) used in AMT input and auxiliary-loss weighting."""
    return tf.stop_gradient(x)


def binary_cross_entropy(y_true, y_pred, name="bce"):
    """Branch-level binary cross-entropy loss (paper Eq. 363)."""
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    loss = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(loss, name=name)


def l2_regularization(scope_names=None):
    """L2 penalty gamma/2 * ||Theta||_2^2 over selected variable scopes."""
    vars_to_reg = []
    for var in tf.compat.v1.trainable_variables():
        if scope_names is None or any(scope in var.name for scope in scope_names):
            vars_to_reg.append(var)
    if not vars_to_reg:
        return 0.0
    return tf.add_n([tf.nn.l2_loss(v) for v in vars_to_reg])


def build_mlp(
    inputs,
    hidden_sizes,
    scope,
    dropout_rate=0.0,
    use_layer_norm=True,
    is_training=True,
    output_activation=None,
):
    """
    Generic ReLU MLP used by SPT and AMT towers.

    The last layer returns a single logit; apply sigmoid outside for y in (0, 1).
    """
    net = inputs
    input_size = inputs.get_shape().as_list()[-1]
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        for i, layer_size in enumerate(hidden_sizes):
            w = tf.compat.v1.get_variable(
                "w_{}".format(i),
                shape=[input_size, layer_size],
                initializer=tf.compat.v1.random_normal_initializer(
                    stddev=1.0 / math.sqrt(float(input_size))
                ),
            )
            b = tf.compat.v1.get_variable(
                "b_{}".format(i),
                shape=[layer_size],
                initializer=tf.compat.v1.zeros_initializer(),
            )
            net = tf.matmul(net, w) + b
            if use_layer_norm:
                net = tf.contrib.layers.layer_norm(net)
            net = tf.nn.relu(net)
            if dropout_rate > 0.0:
                net = tf.compat.v1.layers.dropout(
                    net, rate=dropout_rate, training=is_training
                )
            input_size = layer_size

        # Output logit layer.
        w_out = tf.compat.v1.get_variable(
            "w_out",
            shape=[input_size, 1],
            initializer=tf.compat.v1.random_normal_initializer(
                stddev=1.0 / math.sqrt(float(input_size))
            ),
        )
        b_out = tf.compat.v1.get_variable(
            "b_out", shape=[1], initializer=tf.compat.v1.zeros_initializer()
        )
        logit = tf.matmul(net, w_out) + b_out
        logit = tf.reshape(logit, [-1])

        if output_activation == "sigmoid":
            return tf.nn.sigmoid(logit)
        return logit


class TGSQModule(object):
    """
    Taxonomy-Guided Semantic Quantization (TGSQ).

    Stage 0/1 (offline + pretraining) produces stable semantic IDs. During Stage 2
    joint training, TGSQ parameters are frozen and z_s is consumed as lookup features.
    """

    def __init__(self, config=None):
        self.config = config or cfg.TGSQ_CONFIG
        self.num_levels = self.config["total_semantic_depth"]
        self.sparse_dim = self.config["sparse_sid_dim"]
        self.dense_dim = self.config["dense_sid_dim"]

    def build_sparse_embeddings(self):
        """Trainable sparse SID embeddings E_spr^(m) for Stage-1 pretraining."""
        embeddings = []
        with tf.compat.v1.variable_scope("tgsq_sparse_embeddings", reuse=tf.compat.v1.AUTO_REUSE):
            for level in range(self.num_levels):
                emb = tf.compat.v1.get_variable(
                    "level_{}".format(level),
                    shape=[self.config["codebook_size"], self.sparse_dim],
                    initializer=tf.compat.v1.glorot_uniform_initializer(),
                )
                embeddings.append(emb)
        return embeddings

    def lookup_sparse_view(self, sparse_semantic_ids, sparse_embeddings):
        """
        Sparse hierarchical view e_s (paper Eq. 265).

        Args:
            sparse_semantic_ids: [B, M] int32/int64 indices.
            sparse_embeddings: list of M embedding tables.
        Returns:
            e_s: [B, M * d_spr]
        """
        level_embs = []
        for level, table in enumerate(sparse_embeddings):
            ids = sparse_semantic_ids[:, level]
            level_embs.append(tf.nn.embedding_lookup(table, ids))
        return tf.concat(level_embs, axis=1)

    def fuse_dual_view(self, dense_semantic, sparse_semantic_ids, trainable=True):
        """
        Dual-view fusion z_s = e_d ⊕ e_s (paper Eq. 269).

        In production, dense_semantic is materialized offline. sparse_semantic_ids
        are looked up through E_spr during Stage 1 and frozen afterward.
        """
        with tf.compat.v1.variable_scope("tgsq", reuse=tf.compat.v1.AUTO_REUSE):
            if trainable:
                sparse_tables = self.build_sparse_embeddings()
            else:
                sparse_tables = self.build_sparse_embeddings()
                sparse_tables = [stop_gradient(t) for t in sparse_tables]

            e_s = self.lookup_sparse_view(sparse_semantic_ids, sparse_tables)
            e_d = dense_semantic
            if not trainable:
                e_d = stop_gradient(e_d)
            z_s = tf.concat([e_d, e_s], axis=1, name="z_s")
        return z_s


class MCLFModule(object):
    """
    Monotonicity-Constrained Lifecycle Fusion (MCLF).

    Computes lifecycle stage S(v_cvt), stage embedding h_stg, and gating weight alpha.
    """

    def __init__(self, config=None):
        self.config = config or cfg.MCLF_CONFIG
        self.tau_star = float(self.config["tau_star"])
        self.eta = float(self.config["eta"])
        self.num_stages = int(self.config["num_lifecycle_stages"])
        self.stage_emb_dim = int(self.config["stage_emb_dim"])
        self.num_stats = int(self.config["num_stat_features"])
        self.use_monotonic = bool(self.config["use_monotonic_constraint"])

    def lifecycle_stage_id(self, v_cvt):
        """
        Map conversion count to discrete stage id (paper Eq. 327-334).

        Returns int32 tensor in {0: New, 1: Cold, 2: Growing, 3: Mature}.
        """
        v_cvt = tf.cast(v_cvt, tf.float32)
        new_stage = tf.zeros_like(v_cvt, dtype=tf.int32)
        cold_stage = tf.ones_like(v_cvt, dtype=tf.int32)
        growing_stage = tf.fill(tf.shape(v_cvt), 2)
        mature_stage = tf.fill(tf.shape(v_cvt), 3)

        is_new = tf.equal(v_cvt, 0.0)
        is_cold = tf.logical_and(tf.greater(v_cvt, 0.0), tf.less_equal(v_cvt, self.eta * self.tau_star))
        is_growing = tf.logical_and(
            tf.greater(v_cvt, self.eta * self.tau_star),
            tf.less_equal(v_cvt, self.tau_star),
        )

        stage = tf.where(is_new, new_stage, cold_stage)
        stage = tf.where(is_cold, cold_stage, stage)
        stage = tf.where(is_growing, growing_stage, stage)
        stage = tf.where(tf.greater(v_cvt, self.tau_star), mature_stage, stage)
        return stage

    def stage_embedding(self, stage_ids):
        """Map lifecycle indicator S(v_cvt) to dense vector h_stg."""
        with tf.compat.v1.variable_scope("mclf_stage_embedding", reuse=tf.compat.v1.AUTO_REUSE):
            table = tf.compat.v1.get_variable(
                "stage_table",
                shape=[self.num_stages, self.stage_emb_dim],
                initializer=tf.compat.v1.glorot_uniform_initializer(),
            )
            return tf.nn.embedding_lookup(table, stage_ids)

    def compute_alpha(self, item_stats, v_cvt):
        """
        Monotonic gating weight alpha(h_n, h_stg) (paper Eq. 314-349).

        Args:
            item_stats: [B, 3] posterior counts [clicks, conversions, cost].
            v_cvt: [B] item-level conversion count (used for stage assignment).
        Returns:
            alpha: [B] in (0, 1)
            h_n: [B, 3] log-transformed statistics
            stage_ids: [B] lifecycle stage indices
        """
        item_stats = tf.maximum(item_stats, 0.0)
        h_n = tf.math.log(item_stats + 1.0, name="h_n")
        stage_ids = self.lifecycle_stage_id(v_cvt)
        h_stg = self.stage_embedding(stage_ids)

        with tf.compat.v1.variable_scope("mclf_gate", reuse=tf.compat.v1.AUTO_REUSE):
            w_g = tf.compat.v1.get_variable(
                "W_g",
                shape=[self.stage_emb_dim, self.num_stats],
                initializer=tf.compat.v1.glorot_uniform_initializer(),
            )
            w_b = tf.compat.v1.get_variable(
                "w_b",
                shape=[self.stage_emb_dim],
                initializer=tf.compat.v1.glorot_uniform_initializer(),
            )

            w_alpha_raw = tf.matmul(h_stg, w_g)
            if self.use_monotonic:
                w_alpha = tf.nn.softplus(w_alpha_raw, name="W_alpha")
            else:
                w_alpha = w_alpha_raw

            b_alpha = tf.reduce_sum(h_stg * w_b, axis=1)
            gate_logit = tf.reduce_sum(h_n * w_alpha, axis=1) + b_alpha
            alpha = tf.nn.sigmoid(gate_logit, name="alpha")
        return alpha, h_n, stage_ids


class DFASTModel(object):
    """
    Full D-FAST CVR prediction framework.

    Expected inputs (placeholders or feature tensors):
      - atomic_emb:       flattened x_a embeddings
      - dense_semantic:   TGSQ dense prior view e_d
      - sparse_semantic_ids: [B, M] hierarchical semantic indices
      - user_context_emb: concatenated x_u and x_c
      - item_stats:       [B, 3] item-level posterior statistics
      - labels:           [B] binary conversion labels
    """

    def __init__(
        self,
        tgsq_config=None,
        tower_config=None,
        mclf_config=None,
        loss_config=None,
        tgsq_trainable=False,
        is_training=True,
    ):
        self.tgsq = TGSQModule(tgsq_config)
        self.mclf = MCLFModule(mclf_config)
        self.tower_config = tower_config or cfg.TOWER_CONFIG
        self.loss_config = loss_config or cfg.LOSS_CONFIG
        self.tgsq_trainable = tgsq_trainable
        self.is_training = is_training

    def _build_semantic_prior_tower(self, z_s, user_context_emb):
        """
        Semantic Prior Tower (SPT): masks atomic IDs (paper Eq. 277-285).
        """
        h_p0 = tf.concat([z_s, user_context_emb], axis=1, name="spt_input")
        y_p = build_mlp(
            h_p0,
            hidden_sizes=self.tower_config["hidden_sizes"],
            scope="spt",
            dropout_rate=self.tower_config["dropout_rate"],
            use_layer_norm=self.tower_config["use_layer_norm"],
            is_training=self.is_training,
            output_activation="sigmoid",
        )
        return y_p

    def _build_atomic_memorization_tower(self, atomic_emb, z_s, user_context_emb):
        """
        Atomic Memorization Tower (AMT): uses x_a and sg(z_s) (paper Eq. 290-298).
        """
        h_m0 = tf.concat(
            [atomic_emb, stop_gradient(z_s), user_context_emb],
            axis=1,
            name="amt_input",
        )
        y_m = build_mlp(
            h_m0,
            hidden_sizes=self.tower_config["hidden_sizes"],
            scope="amt",
            dropout_rate=self.tower_config["dropout_rate"],
            use_layer_norm=self.tower_config["use_layer_norm"],
            is_training=self.is_training,
            output_activation="sigmoid",
        )
        return y_m

    def forward(
        self,
        atomic_emb,
        dense_semantic,
        sparse_semantic_ids,
        user_context_emb,
        item_stats,
    ):
        """
        Forward pass returning fused prediction and intermediate tower outputs.
        """
        z_s = self.tgsq.fuse_dual_view(
            dense_semantic,
            sparse_semantic_ids,
            trainable=self.tgsq_trainable,
        )

        y_p = self._build_semantic_prior_tower(z_s, user_context_emb)
        y_m = self._build_atomic_memorization_tower(atomic_emb, z_s, user_context_emb)

        v_cvt = item_stats[:, 1]
        alpha, h_n, stage_ids = self.mclf.compute_alpha(item_stats, v_cvt)

        # Dynamic fusion (paper Eq. 304).
        y_f = alpha * y_m + (1.0 - alpha) * y_p

        outputs = {
            "y_f": y_f,
            "y_p": y_p,
            "y_m": y_m,
            "alpha": alpha,
            "z_s": z_s,
            "h_n": h_n,
            "stage_ids": stage_ids,
        }
        return outputs

    def compute_loss(self, outputs, labels):
        """
        Unified optimization objective L_tot (paper Eq. 367).
        """
        labels = tf.cast(labels, tf.float32)
        y_f = outputs["y_f"]
        y_p = outputs["y_p"]
        y_m = outputs["y_m"]
        alpha = outputs["alpha"]

        loss_f = binary_cross_entropy(labels, y_f, name="loss_f")
        loss_p = binary_cross_entropy(labels, y_p, name="loss_p")
        loss_m = binary_cross_entropy(labels, y_m, name="loss_m")

        aux_weight = float(self.loss_config["aux_loss_weight"])
        spt_scale = (1.0 - stop_gradient(alpha))
        aux_loss = spt_scale * loss_p + loss_m

        reg = l2_regularization(scope_names=["spt", "amt", "mclf", "tgsq"])
        reg_weight = float(self.loss_config["l2_reg_weight"])

        total_loss = loss_f + aux_weight * aux_loss + 0.5 * reg_weight * reg
        total_loss = tf.identity(total_loss, name="loss_total")

        loss_dict = {
            "loss_total": total_loss,
            "loss_f": loss_f,
            "loss_p": loss_p,
            "loss_m": loss_m,
            "loss_aux": aux_loss,
            "loss_reg": reg,
        }
        return loss_dict

    def build_graph(
        self,
        atomic_emb,
        dense_semantic,
        sparse_semantic_ids,
        user_context_emb,
        item_stats,
        labels,
    ):
        """Build forward pass and loss; returns (outputs, loss_dict)."""
        outputs = self.forward(
            atomic_emb=atomic_emb,
            dense_semantic=dense_semantic,
            sparse_semantic_ids=sparse_semantic_ids,
            user_context_emb=user_context_emb,
            item_stats=item_stats,
        )
        loss_dict = self.compute_loss(outputs, labels)
        return outputs, loss_dict


def build_placeholders(batch_size=None):
    """
    Utility placeholders for standalone testing or dataset wiring.

    Dimensions follow the paper's experimental configuration where applicable.
    """
    tgsq_cfg = cfg.TGSQ_CONFIG
    atomic_fields = sum(x["num_fields"] for x in cfg.ATOMIC_SPARSE_FEAT_CONFIGS)
    atomic_dim = atomic_fields * cfg.TOWER_CONFIG["atomic_emb_dim"]

    user_fields = sum(x["num_fields"] for x in cfg.USER_CONTEXT_FEAT_CONFIGS)
    user_context_dim = user_fields * cfg.TOWER_CONFIG["atomic_emb_dim"]

    dense_dim = tgsq_cfg["dense_sid_dim"]
    num_levels = tgsq_cfg["total_semantic_depth"]

    if batch_size is None:
        shape_prefix = [None]
    else:
        shape_prefix = [batch_size]

    placeholders = {
        "atomic_emb": tf.compat.v1.placeholder(
            tf.float32, shape_prefix + [atomic_dim], name="atomic_emb"
        ),
        "dense_semantic": tf.compat.v1.placeholder(
            tf.float32, shape_prefix + [dense_dim], name="dense_semantic"
        ),
        "sparse_semantic_ids": tf.compat.v1.placeholder(
            tf.int32, shape_prefix + [num_levels], name="sparse_semantic_ids"
        ),
        "user_context_emb": tf.compat.v1.placeholder(
            tf.float32, shape_prefix + [user_context_dim], name="user_context_emb"
        ),
        "item_stats": tf.compat.v1.placeholder(
            tf.float32, shape_prefix + [3], name="item_stats"
        ),
        "labels": tf.compat.v1.placeholder(
            tf.float32, shape_prefix, name="labels"
        ),
    }
    return placeholders


def build_training_graph(
    batch_size=None,
    tgsq_trainable=False,
    is_training=True,
):
    """
    End-to-end training graph for Stage-2 joint D-FAST optimization.

    Example usage:
        placeholders, outputs, loss_dict = build_training_graph()
        train_op = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(loss_dict["loss_total"])
    """
    placeholders = build_placeholders(batch_size=batch_size)
    model = DFASTModel(tgsq_trainable=tgsq_trainable, is_training=is_training)
    outputs, loss_dict = model.build_graph(
        atomic_emb=placeholders["atomic_emb"],
        dense_semantic=placeholders["dense_semantic"],
        sparse_semantic_ids=placeholders["sparse_semantic_ids"],
        user_context_emb=placeholders["user_context_emb"],
        item_stats=placeholders["item_stats"],
        labels=placeholders["labels"],
    )
    return placeholders, outputs, loss_dict
