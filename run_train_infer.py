#!/usr/bin/env python3
"""
================================================================================
Nanobody Phage Display Deep Learning Pipeline
================================================================================

End-to-end multi-task deep learning pipeline for nanobody phage display
library analysis. Trains from scratch on FASTA files from selection rounds
(R0, R1, R3) with no pretrained models, no IMGT numbering, and no
pre-clustering.

Architecture Overview:
    Stage 1 - Masked Language Model (MLM) pretraining of sequence encoder
        Transformer encoder learns amino acid grammar of the library via
        masked token prediction on all sequences (ignores enrichment).

    Stage 2 - Multi-task enrichment model
        Sequence encoder (from Stage 1) -> latent decomposition into
        z_bind (binding signal), z_bio (biophysical/amplification), and
        z_family (functional clustering). Jointly trains:
            - Enrichment trajectory prediction (MSE)
            - Amplification bias prediction from z_bio (MSE)
            - Gradient reversal disentanglement on z_bind (adversarial MSE)
            - Self-supervised contrastive clustering on z_family (NT-Xent)

Data Format:
    Three FASTA files (one per round). Headers: >identifier_counts
    Identifiers are mappable across rounds. Missing sequences in R1/R3
    have count 0. Missing sequences in R0 have count 0.5 (pseudocount).

Usage:
    # Stage 1: MLM pretraining
    python nanobody_pipeline.py --stage mlm \\
        --r0 round0.fasta --r1 round1.fasta --r3 round3.fasta \\
        --epochs 20 --batch_size 256

    # Stage 2: Multi-task training (loads MLM weights)
    python nanobody_pipeline.py --stage multitask \\
        --r0 round0.fasta --r1 round1.fasta --r3 round3.fasta \\
        --mlm_weights checkpoints/mlm_best.weights.h5 \\
        --epochs 50 --batch_size 256

    # Inference
    python nanobody_pipeline.py --stage inference \\
        --r0 round0.fasta --r1 round1.fasta --r3 round3.fasta \\
        --multitask_weights checkpoints/multitask_best.weights.h5 \\
        --output_prefix results/predictions

Author: Generated via Claude / Anthropic
Date: 2026-02-15
================================================================================
"""

import os
import sys
import math
import time
import random
import logging
import argparse
import numpy as np
import tensorflow as tf
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}
PAD_IDX = 20   # padding token index
MASK_IDX = 21  # mask token index for MLM
UNK_IDX = 22   # unknown amino acid token
VOCAB_SIZE = 23  # 20 AA + pad + mask + unk

# BLOSUM62 substitution matrix (20x20, standard amino acid order ACDEFGHIKLMNPQRSTVWY)
# Each row is the substitution score profile for that amino acid.
# Used as fixed initial embedding: each AA maps to its 20-dim BLOSUM62 row.
BLOSUM62_MATRIX = np.array([
    # A   C   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   T   V   W   Y
    [ 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2],  # A
    [ 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],  # C
    [-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3],  # D
    [-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2],  # E
    [-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3],  # F
    [ 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3],  # G
    [-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2],  # H
    [-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1],  # I
    [-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2],  # K
    [-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1],  # L
    [-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1],  # M
    [-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2],  # N
    [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3],  # P
    [-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1],  # Q
    [-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2],  # R
    [ 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2],  # S
    [ 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2],  # T
    [ 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1],  # V
    [-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2],  # W
    [-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7],  # Y
], dtype=np.float32)

# Normalize BLOSUM62 rows to zero mean, unit variance for better gradient flow
_blosum_mean = BLOSUM62_MATRIX.mean(axis=1, keepdims=True)
_blosum_std = BLOSUM62_MATRIX.std(axis=1, keepdims=True) + 1e-8
BLOSUM62_NORMED = (BLOSUM62_MATRIX - _blosum_mean) / _blosum_std

# Build full embedding table: 20 AA rows + pad(zeros) + mask(zeros) + unk(zeros)
BLOSUM62_EMBED = np.vstack([
    BLOSUM62_NORMED,
    np.zeros((1, 20), dtype=np.float32),  # PAD
    np.zeros((1, 20), dtype=np.float32),  # MASK
    np.zeros((1, 20), dtype=np.float32),  # UNK
])  # shape: [23, 20]


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

def parse_args():
    """Parse command-line arguments for all pipeline stages and hyperparameters.

    Returns:
        argparse.Namespace: Parsed arguments with all configuration values.
    """
    p = argparse.ArgumentParser(
        description="Nanobody Phage Display Deep Learning Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Stage selection ---
    p.add_argument("--stage", type=str, required=True, choices=["mlm", "multitask", "inference"],
                   help="Pipeline stage: 'mlm' for pretraining, 'multitask' for full training, 'inference' for prediction.")
    # --- Data paths ---
    p.add_argument("--r0", type=str, required=True, help="Path to Round 0 FASTA file.")
    p.add_argument("--r1", type=str, required=True, help="Path to Round 1 FASTA file.")
    p.add_argument("--r3", type=str, required=True, help="Path to Round 3 FASTA file.")
    # --- Weight loading ---
    p.add_argument("--mlm_weights", type=str, default=None, help="Path to pretrained MLM encoder weights (.weights.h5). Required for multitask stage.")
    p.add_argument("--multitask_weights", type=str, default=None, help="Path to trained multitask model weights (.weights.h5). Required for inference stage.")
    # --- Output ---
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    p.add_argument("--output_prefix", type=str, default="results/predictions", help="Output prefix for inference results (inference stage).")
    # --- Data splitting ---
    p.add_argument("--split", action="store_true", default=True, help="Split data into train/val/test sets.")
    p.add_argument("--no_split", action="store_false", dest="split", help="Use all data for training (no split).")
    p.add_argument("--val_frac", type=float, default=0.05, help="Fraction of data for validation set.")
    p.add_argument("--test_frac", type=float, default=0.05, help="Fraction of data for test set.")
    # --- Sequence parameters ---
    p.add_argument("--max_seq_len", type=int, default=150, help="Maximum sequence length (right-padded). Sequences longer than this are truncated.")
    p.add_argument("--pseudocount", type=float, default=0.5, help="Pseudocount added to zero-count frequencies before log transform.")
    # --- Architecture: encoder ---
    p.add_argument("--d_model", type=int, default=128, help="Transformer model dimension throughout the encoder.")
    p.add_argument("--n_heads", type=int, default=4, help="Number of attention heads in transformer layers.")
    p.add_argument("--n_encoder_layers", type=int, default=4, help="Number of transformer encoder layers in the sequence encoder.")
    p.add_argument("--ff_dim", type=int, default=256, help="Feed-forward inner dimension in transformer blocks.")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate applied in transformer layers and heads.")
    p.add_argument("--pos_encoding", type=str, default="learned", choices=["learned", "sinusoidal"],
                   help="Positional encoding type for the sequence encoder.")
    # --- Architecture: latent space ---
    p.add_argument("--z_bind_dim", type=int, default=32, help="Dimensionality of binding latent subspace (z_bind).")
    p.add_argument("--z_bio_dim", type=int, default=32, help="Dimensionality of biophysical latent subspace (z_bio).")
    p.add_argument("--z_family_dim", type=int, default=32, help="Dimensionality of family/clustering latent subspace (z_family).")
    # --- Architecture: temporal ---
    p.add_argument("--n_temporal_layers", type=int, default=2, help="Number of transformer layers in the temporal enrichment module.")
    p.add_argument("--n_round_embed_dim", type=int, default=16, help="Dimensionality of learned round identity embeddings.")
    # --- MLM parameters ---
    p.add_argument("--mlm_mask_frac", type=float, default=0.15, help="Fraction of tokens to mask during MLM pretraining.")
    # --- Loss weights ---
    p.add_argument("--w_trajectory", type=float, default=1.0, help="Weight for enrichment trajectory prediction loss (MSE).")
    p.add_argument("--w_amplification", type=float, default=0.5, help="Weight for amplification bias prediction loss from z_bio (MSE).")
    p.add_argument("--w_grl", type=float, default=0.3, help="Weight for gradient reversal disentanglement loss on z_bind.")
    p.add_argument("--w_contrastive", type=float, default=0.5, help="Weight for NT-Xent contrastive loss on z_family.")
    p.add_argument("--w_mlm_aux", type=float, default=0.1, help="Weight for auxiliary MLM loss during multitask training (0 to disable).")
    p.add_argument("--contrastive_temp", type=float, default=0.1, help="Temperature for NT-Xent contrastive loss.")
    p.add_argument("--n_mutations", type=int, default=2, help="Number of random AA mutations for contrastive augmentation.")
    p.add_argument("--grl_lambda", type=float, default=1.0, help="Gradient reversal layer scaling factor (lambda).")
    # --- Training ---
    p.add_argument("--epochs", type=int, default=50, help="Maximum number of training epochs.")
    p.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    p.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd", "rmsprop"],
                   help="Optimizer for training.")
    p.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for AdamW optimizer.")
    p.add_argument("--lr_schedule", type=str, default="cosine", choices=["constant", "cosine", "warmup_cosine"],
                   help="Learning rate schedule.")
    p.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps for warmup_cosine schedule.")
    p.add_argument("--convergence", action="store_true", default=False,
                   help="Enable convergence-based early stopping (monitors val loss plateau).")
    p.add_argument("--patience", type=int, default=5, help="Epochs of no val loss improvement before early stopping (requires --convergence).")
    p.add_argument("--min_delta", type=float, default=1e-4, help="Minimum loss improvement to reset patience counter.")
    p.add_argument("--mixed_precision", action="store_true", default=False, help="Enable mixed precision (float16) training for GPU speedup.")
    # --- Reproducibility ---
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    # --- Inference ---
    p.add_argument("--inference_batch_size", type=int, default=512, help="Batch size for inference (can be larger than training).")
    return p.parse_args()


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def set_seeds(seed):
    """Set all random seeds for reproducibility.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================

def parse_fasta_streaming(filepath):
    """Parse a FASTA file yielding (identifier, count, sequence) tuples.

    Reads the file line by line for memory efficiency. Headers must follow
    the format: >identifier_count where count is a numeric value.

    Args:
        filepath (str): Path to FASTA file.

    Yields:
        tuple: (identifier: str, count: float, sequence: str)
    """
    identifier, count, seq_lines = None, None, []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    yield identifier, count, "".join(seq_lines)
                header = line[1:]  # strip '>'
                parts = header.rsplit("_", 1)  # split on LAST underscore
                identifier = parts[0]
                count = float(parts[1])
                seq_lines = []
            else:
                seq_lines.append(line.upper())
    if identifier is not None:
        yield identifier, count, "".join(seq_lines)


def load_and_merge_rounds(r0_path, r1_path, r3_path, max_seq_len, pseudocount=0.5):
    """Load three FASTA round files and merge by sequence identifier.

    Computes normalized log-frequencies and detection flags per round.
    Returns numpy arrays ready for dataset construction.

    Args:
        r0_path (str): Path to Round 0 FASTA file.
        r1_path (str): Path to Round 1 FASTA file.
        r3_path (str): Path to Round 3 FASTA file.
        max_seq_len (int): Maximum sequence length (truncate/pad to this).
        pseudocount (float): Pseudocount for zero-count log frequency.

    Returns:
        dict with keys:
            'seq_indices' : np.ndarray, shape [N, max_seq_len], dtype int8
                Amino acid index encoded sequences (0-19=AA, 20=pad, 22=unk).
            'seq_lengths' : np.ndarray, shape [N], dtype int16
                Actual sequence length before padding.
            'counts'      : np.ndarray, shape [N, 3], dtype float32
                Raw counts [R0, R1, R3].
            'log_freqs'   : np.ndarray, shape [N, 3], dtype float32
                Log2-transformed normalized frequencies [R0, R1, R3].
            'detected'    : np.ndarray, shape [N, 3], dtype float32
                Detection flags (1.0 if genuinely detected, 0.0 if imputed).
            'ids'         : list of str
                Sequence identifiers in order.
            'total_counts': np.ndarray, shape [3], dtype float64
                Total read counts per round for normalization reference.
    """
    logger.info("Loading Round 0: %s", r0_path)
    r0_data = {sid: (cnt, seq) for sid, cnt, seq in parse_fasta_streaming(r0_path)}
    logger.info("  -> %d sequences", len(r0_data))
    logger.info("Loading Round 1: %s", r1_path)
    r1_counts = {sid: cnt for sid, cnt, _ in parse_fasta_streaming(r1_path)}
    logger.info("  -> %d sequences", len(r1_counts))
    logger.info("Loading Round 3: %s", r3_path)
    r3_counts = {sid: cnt for sid, cnt, _ in parse_fasta_streaming(r3_path)}
    logger.info("  -> %d sequences", len(r3_counts))
    # Collect all unique identifiers. R0 should contain all (user added missing with 0.5).
    all_ids = sorted(r0_data.keys())
    n = len(all_ids)
    logger.info("Total unique sequences: %d", n)
    # Compute total counts per round (for frequency normalization)
    total_r0 = sum(cnt for cnt, _ in r0_data.values())
    total_r1 = sum(r1_counts.values())
    total_r3 = sum(r3_counts.values())
    total_counts = np.array([total_r0, total_r1, total_r3], dtype=np.float64)
    logger.info("Total counts per round: R0=%.0f, R1=%.0f, R3=%.0f", total_r0, total_r1, total_r3)
    # Preallocate arrays
    seq_indices = np.full((n, max_seq_len), PAD_IDX, dtype=np.int8)  # [N, L]
    seq_lengths = np.zeros(n, dtype=np.int16)  # [N]
    counts = np.zeros((n, 3), dtype=np.float32)  # [N, 3]
    detected = np.zeros((n, 3), dtype=np.float32)  # [N, 3]
    # Encode sequences and fill counts
    for i, sid in enumerate(all_ids):
        cnt_r0, seq = r0_data[sid]
        cnt_r1 = r1_counts.get(sid, 0.0)
        cnt_r3 = r3_counts.get(sid, 0.0)
        # Encode sequence to integer indices
        L = min(len(seq), max_seq_len)
        seq_lengths[i] = L
        for j in range(L):
            aa = seq[j]
            seq_indices[i, j] = AA_TO_IDX.get(aa, UNK_IDX)
        # Raw counts
        counts[i] = [cnt_r0, cnt_r1, cnt_r3]
        # Detection flags: R0 > pseudocount means genuinely sequenced
        detected[i, 0] = 1.0 if cnt_r0 > pseudocount else 0.0
        detected[i, 1] = 1.0 if cnt_r1 > 0 else 0.0
        detected[i, 2] = 1.0 if cnt_r3 > 0 else 0.0
        if (i + 1) % 1_000_000 == 0:
            logger.info("  Processed %d / %d sequences", i + 1, n)
    # Compute log2 frequencies with pseudocount
    freqs = counts / total_counts[np.newaxis, :].astype(np.float32)  # [N, 3]
    log_freqs = np.log2(freqs + pseudocount / total_counts[np.newaxis, :].astype(np.float32))  # [N, 3]
    # Standardize log_freqs per round (zero mean, unit variance) for stable training
    lf_mean = log_freqs.mean(axis=0, keepdims=True)  # [1, 3]
    lf_std = log_freqs.std(axis=0, keepdims=True) + 1e-8  # [1, 3]
    log_freqs_normed = ((log_freqs - lf_mean) / lf_std).astype(np.float32)  # [N, 3]
    logger.info("Log-freq stats (raw): mean=%s, std=%s", lf_mean.flatten(), lf_std.flatten())
    # Free memory
    del r0_data, r1_counts, r3_counts
    return {
        "seq_indices": seq_indices,       # [N, max_seq_len] int8
        "seq_lengths": seq_lengths,       # [N] int16
        "counts": counts,                 # [N, 3] float32
        "log_freqs": log_freqs_normed,    # [N, 3] float32 (standardized)
        "log_freqs_raw": log_freqs,       # [N, 3] float32 (raw log2 freqs, for reference)
        "log_freq_mean": lf_mean,         # [1, 3] for de-standardization
        "log_freq_std": lf_std,           # [1, 3] for de-standardization
        "detected": detected,             # [N, 3] float32
        "ids": all_ids,                   # list[str]
        "total_counts": total_counts,     # [3] float64
    }


def split_indices(n, val_frac, test_frac, seed):
    """Generate train/val/test index splits with shuffled random assignment.

    Args:
        n (int): Total number of samples.
        val_frac (float): Fraction allocated to validation.
        test_frac (float): Fraction allocated to test.
        seed (int): Random seed for reproducible splitting.

    Returns:
        tuple: (train_idx, val_idx, test_idx) as np.ndarrays of indices.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    return train_idx, val_idx, test_idx


# ==============================================================================
# TENSORFLOW DATASET CONSTRUCTION
# ==============================================================================

def build_mlm_dataset(seq_indices, seq_lengths, indices, batch_size, mask_frac, seed, shuffle=True):
    """Build a tf.data.Dataset for MLM pretraining with on-the-fly masking.

    Each element yields (input_ids, mask_positions, mask_labels, padding_mask).
    Masking follows BERT convention: 80% [MASK], 10% random, 10% keep.

    Args:
        seq_indices (np.ndarray): [N, L] integer-encoded sequences.
        seq_lengths (np.ndarray): [N] true sequence lengths.
        indices (np.ndarray): Subset indices to include.
        batch_size (int): Batch size.
        mask_frac (float): Fraction of non-pad tokens to mask.
        seed (int): Random seed.
        shuffle (bool): Whether to shuffle each epoch.

    Returns:
        tf.data.Dataset: Yields batched dicts with keys:
            'input_ids'    : [B, L] int32 - tokens with masking applied
            'original_ids' : [B, L] int32 - original tokens before masking
            'mask_flag'    : [B, L] float32 - 1.0 at masked positions, 0.0 elsewhere
            'padding_mask' : [B, L] float32 - 1.0 at real tokens, 0.0 at padding
    """
    sub_seqs = seq_indices[indices]    # [n_sub, L]
    sub_lens = seq_lengths[indices]    # [n_sub]
    max_len = sub_seqs.shape[1]
    n_sub = len(indices)
    def generator():
        rng = np.random.RandomState(seed)
        order = np.arange(n_sub)
        while True:  # infinite generator, use .take() or steps_per_epoch
            if shuffle:
                rng.shuffle(order)
            for i in order:
                seq = sub_seqs[i].copy().astype(np.int32)  # [L]
                length = int(sub_lens[i])
                original = seq.copy()
                pad_mask = np.zeros(max_len, dtype=np.float32)
                pad_mask[:length] = 1.0
                mask_flag = np.zeros(max_len, dtype=np.float32)
                # Select positions to mask (only non-pad)
                n_mask = max(1, int(length * mask_frac))
                positions = rng.choice(length, size=n_mask, replace=False)
                for pos in positions:
                    mask_flag[pos] = 1.0
                    r = rng.random()
                    if r < 0.8:
                        seq[pos] = MASK_IDX      # 80% -> [MASK]
                    elif r < 0.9:
                        seq[pos] = rng.randint(0, 20)  # 10% -> random AA
                    # else 10% -> keep original
                yield {"input_ids": seq, "original_ids": original, "mask_flag": mask_flag, "padding_mask": pad_mask}
    output_sig = {
        "input_ids": tf.TensorSpec([max_len], tf.int32),
        "original_ids": tf.TensorSpec([max_len], tf.int32),
        "mask_flag": tf.TensorSpec([max_len], tf.float32),
        "padding_mask": tf.TensorSpec([max_len], tf.float32),
    }
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_sig)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, n_sub


def build_multitask_dataset(seq_indices, seq_lengths, log_freqs, detected, indices, batch_size, seed, shuffle=True, mask_frac=0.15, n_mutations=2):
    """Build a tf.data.Dataset for multi-task enrichment training.

    Each element provides sequence data, enrichment targets, and augmented
    sequences for contrastive learning.

    Args:
        seq_indices (np.ndarray): [N, L] integer-encoded sequences.
        seq_lengths (np.ndarray): [N] true sequence lengths.
        log_freqs (np.ndarray): [N, 3] standardized log-frequencies (targets).
        detected (np.ndarray): [N, 3] detection flags.
        indices (np.ndarray): Subset indices to include.
        batch_size (int): Batch size.
        seed (int): Random seed.
        shuffle (bool): Whether to shuffle each epoch.
        mask_frac (float): MLM mask fraction for auxiliary MLM loss.
        n_mutations (int): Number of random mutations for contrastive augmentation.

    Returns:
        tf.data.Dataset: Yields batched dicts with keys:
            'input_ids'     : [B, L] int32 - original sequence tokens
            'mlm_input_ids' : [B, L] int32 - masked sequence (for aux MLM)
            'mlm_mask_flag' : [B, L] float32 - mask positions for aux MLM
            'aug_input_ids' : [B, L] int32 - mutated sequence (for contrastive)
            'padding_mask'  : [B, L] float32 - 1 at real tokens, 0 at pad
            'log_freqs'     : [B, 3] float32 - target log frequencies
            'detected'      : [B, 3] float32 - detection flags
    """
    sub_seqs = seq_indices[indices]
    sub_lens = seq_lengths[indices]
    sub_lf = log_freqs[indices]
    sub_det = detected[indices]
    max_len = sub_seqs.shape[1]
    n_sub = len(indices)
    def generator():
        rng = np.random.RandomState(seed)
        order = np.arange(n_sub)
        while True:
            if shuffle:
                rng.shuffle(order)
            for i in order:
                seq = sub_seqs[i].astype(np.int32)  # [L]
                length = int(sub_lens[i])
                lf = sub_lf[i].astype(np.float32)   # [3]
                det = sub_det[i].astype(np.float32)  # [3]
                pad_mask = np.zeros(max_len, dtype=np.float32)
                pad_mask[:length] = 1.0
                # --- MLM augmentation for auxiliary loss ---
                mlm_seq = seq.copy()
                mlm_flag = np.zeros(max_len, dtype=np.float32)
                n_mask = max(1, int(length * mask_frac))
                mlm_positions = rng.choice(length, size=n_mask, replace=False)
                for pos in mlm_positions:
                    mlm_flag[pos] = 1.0
                    r = rng.random()
                    if r < 0.8:
                        mlm_seq[pos] = MASK_IDX
                    elif r < 0.9:
                        mlm_seq[pos] = rng.randint(0, 20)
                # --- Contrastive augmentation (random AA mutations) ---
                aug_seq = seq.copy()
                if length > 0:
                    n_mut = min(n_mutations, length)
                    mut_positions = rng.choice(length, size=n_mut, replace=False)
                    for pos in mut_positions:
                        new_aa = rng.randint(0, 20)
                        while new_aa == seq[pos]:
                            new_aa = rng.randint(0, 20)
                        aug_seq[pos] = new_aa
                yield {
                    "input_ids": seq, "mlm_input_ids": mlm_seq, "mlm_mask_flag": mlm_flag,
                    "aug_input_ids": aug_seq, "padding_mask": pad_mask,
                    "log_freqs": lf, "detected": det,
                }
    output_sig = {
        "input_ids": tf.TensorSpec([max_len], tf.int32),
        "mlm_input_ids": tf.TensorSpec([max_len], tf.int32),
        "mlm_mask_flag": tf.TensorSpec([max_len], tf.float32),
        "aug_input_ids": tf.TensorSpec([max_len], tf.int32),
        "padding_mask": tf.TensorSpec([max_len], tf.float32),
        "log_freqs": tf.TensorSpec([3], tf.float32),
        "detected": tf.TensorSpec([3], tf.float32),
    }
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_sig)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, n_sub


# ==============================================================================
# CUSTOM LAYERS
# ==============================================================================

class BLOSUM62Embedding(tf.keras.layers.Layer):
    """Fixed BLOSUM62-based amino acid embedding lookup.

    Maps integer token indices to their corresponding 20-dimensional
    BLOSUM62 substitution score vectors. The embedding is NOT trainable;
    it provides a biochemically informed initialization. A subsequent
    trainable linear projection maps to d_model dimensions.

    Input:  [batch, seq_len] int32 token indices (0-22)
    Output: [batch, seq_len, d_model] float32 embeddings
    """
    def __init__(self, d_model, **kwargs):
        """
        Args:
            d_model (int): Output embedding dimensionality.
        """
        super().__init__(**kwargs)
        self.d_model = d_model
        self.blosum_table = tf.constant(BLOSUM62_EMBED, dtype=tf.float32)  # [23, 20]
        self.projection = tf.keras.layers.Dense(d_model, use_bias=True, name="blosum_proj")  # [20] -> [d_model]

    def call(self, token_ids):
        """
        Args:
            token_ids: [batch, seq_len] int32

        Returns:
            [batch, seq_len, d_model] float32
        """
        blosum_emb = tf.gather(self.blosum_table, token_ids)  # [B, L, 20]
        return self.projection(blosum_emb)  # [B, L, d_model]


class LearnedPositionalEncoding(tf.keras.layers.Layer):
    """Learned absolute positional embeddings added to token embeddings.

    Input:  [batch, seq_len, d_model]
    Output: [batch, seq_len, d_model] (with positional info added)
    """
    def __init__(self, max_len, d_model, **kwargs):
        """
        Args:
            max_len (int): Maximum sequence length supported.
            d_model (int): Embedding dimensionality (must match input).
        """
        super().__init__(**kwargs)
        self.pos_emb = self.add_weight(name="pos_emb", shape=[max_len, d_model], initializer="glorot_uniform", trainable=True)  # [L, d_model]

    def call(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        seq_len = tf.shape(x)[1]
        return x + self.pos_emb[tf.newaxis, :seq_len, :]  # [B, L, d_model]


class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017).

    Input:  [batch, seq_len, d_model]
    Output: [batch, seq_len, d_model]
    """
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(max_len)[:, np.newaxis]  # [L, 1]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)  # [1, L, d_model]

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]  # [B, L, d_model]


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Single transformer encoder block with multi-head self-attention and FFN.

    Pre-norm architecture (LayerNorm before attention/FFN) for stable training.

    Input:  [batch, seq_len, d_model], padding_mask [batch, seq_len]
    Output: [batch, seq_len, d_model]
    """
    def __init__(self, d_model, n_heads, ff_dim, dropout, **kwargs):
        """
        Args:
            d_model (int): Model dimension.
            n_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward inner dimension.
            dropout (float): Dropout rate.
        """
        super().__init__(**kwargs)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="gelu"),  # [B, L, ff_dim]
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),  # [B, L, d_model]
            tf.keras.layers.Dropout(dropout),
        ])

    def call(self, x, padding_mask=None, training=False):
        """
        Args:
            x: [batch, seq_len, d_model]
            padding_mask: [batch, seq_len] float32 (1=real, 0=pad)
            training: bool

        Returns:
            [batch, seq_len, d_model]
        """
        # Convert padding mask to attention mask: [B, L] -> [B, 1, 1, L] where 0=attend, large_neg=ignore
        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]  # [B, 1, 1, L]
            attn_mask = (1.0 - attn_mask) * -1e9  # invert: 0->0, pad->-1e9
        normed = self.ln1(x)  # [B, L, d_model]
        attn_out = self.mha(normed, normed, attention_mask=attn_mask, training=training)  # [B, L, d_model]
        x = x + self.drop1(attn_out, training=training)  # [B, L, d_model]
        normed2 = self.ln2(x)  # [B, L, d_model]
        ffn_out = self.ffn(normed2, training=training)  # [B, L, d_model]
        return x + ffn_out  # [B, L, d_model]


class SequenceEncoder(tf.keras.layers.Layer):
    """Full sequence encoder: BLOSUM62 embedding + positional encoding + transformer stack.

    Encodes an amino acid sequence into contextual per-residue representations,
    then pools to a single sequence-level vector.

    Input:  token_ids [batch, seq_len] int32, padding_mask [batch, seq_len] float32
    Output: pooled [batch, d_model], per_residue [batch, seq_len, d_model]
    """
    def __init__(self, d_model, n_heads, n_layers, ff_dim, dropout, max_len, pos_encoding="learned", **kwargs):
        """
        Args:
            d_model (int): Model dimension.
            n_heads (int): Attention heads.
            n_layers (int): Number of transformer encoder layers.
            ff_dim (int): FFN inner dimension.
            dropout (float): Dropout rate.
            max_len (int): Maximum sequence length.
            pos_encoding (str): 'learned' or 'sinusoidal'.
        """
        super().__init__(**kwargs)
        self.embedding = BLOSUM62Embedding(d_model, name="blosum_embed")
        if pos_encoding == "learned":
            self.pos_enc = LearnedPositionalEncoding(max_len, d_model, name="pos_enc")
        else:
            self.pos_enc = SinusoidalPositionalEncoding(max_len, d_model, name="pos_enc")
        self.embed_dropout = tf.keras.layers.Dropout(dropout)
        self.encoder_layers = [
            TransformerEncoderBlock(d_model, n_heads, ff_dim, dropout, name=f"enc_block_{i}")
            for i in range(n_layers)
        ]
        self.final_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, token_ids, padding_mask, training=False):
        """
        Args:
            token_ids:    [batch, seq_len] int32
            padding_mask: [batch, seq_len] float32 (1=real, 0=pad)
            training:     bool

        Returns:
            pooled:       [batch, d_model] - mean-pooled sequence embedding
            per_residue:  [batch, seq_len, d_model] - per-position embeddings
        """
        x = self.embedding(token_ids)  # [B, L, d_model]
        x = self.pos_enc(x)  # [B, L, d_model]
        x = self.embed_dropout(x, training=training)  # [B, L, d_model]
        for layer in self.encoder_layers:
            x = layer(x, padding_mask=padding_mask, training=training)  # [B, L, d_model]
        x = self.final_ln(x)  # [B, L, d_model]
        # Mean pool over non-pad positions
        mask_expanded = padding_mask[:, :, tf.newaxis]  # [B, L, 1]
        pooled = tf.reduce_sum(x * mask_expanded, axis=1) / (tf.reduce_sum(mask_expanded, axis=1) + 1e-8)  # [B, d_model]
        return pooled, x


class TemporalEnrichmentModule(tf.keras.layers.Layer):
    """Temporal transformer over selection rounds.

    Constructs per-round tokens from the pooled sequence embedding, round
    identity, log-frequency, and detection flag, then applies self-attention
    across rounds to model enrichment dynamics.

    Input:  seq_emb [B, d_model], log_freqs [B, 3], detected [B, 3]
    Output: temporal_emb [B, d_model] (pooled over rounds)
    """
    def __init__(self, d_model, n_heads, n_layers, ff_dim, dropout, n_rounds=3, round_embed_dim=16, **kwargs):
        """
        Args:
            d_model (int): Model dimension.
            n_heads (int): Attention heads for temporal attention.
            n_layers (int): Number of temporal transformer layers.
            ff_dim (int): FFN inner dimension.
            dropout (float): Dropout rate.
            n_rounds (int): Number of selection rounds (3 for R0, R1, R3).
            round_embed_dim (int): Dimension of learned round embeddings.
        """
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.round_embeddings = self.add_weight(name="round_emb", shape=[n_rounds, round_embed_dim], initializer="glorot_uniform", trainable=True)  # [3, round_embed_dim]
        # Input per round token: d_model (seq_emb) + round_embed_dim + 1 (log_freq) + 1 (detected)
        token_input_dim = d_model + round_embed_dim + 2
        self.token_proj = tf.keras.layers.Dense(d_model, activation="gelu", name="round_token_proj")  # [token_input_dim] -> [d_model]
        self.temporal_layers = [
            TransformerEncoderBlock(d_model, n_heads, ff_dim, dropout, name=f"temporal_block_{i}")
            for i in range(n_layers)
        ]
        self.final_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, seq_emb, log_freqs, detected, training=False):
        """
        Args:
            seq_emb:  [batch, d_model] pooled sequence embedding
            log_freqs: [batch, 3] standardized log frequencies per round
            detected:  [batch, 3] detection flags per round

        Returns:
            temporal_pooled: [batch, d_model] mean-pooled temporal embedding
        """
        B = tf.shape(seq_emb)[0]
        tokens = []
        for r in range(self.n_rounds):
            round_emb = tf.tile(self.round_embeddings[r:r+1, :], [B, 1])  # [B, round_embed_dim]
            lf = log_freqs[:, r:r+1]  # [B, 1]
            det = detected[:, r:r+1]  # [B, 1]
            token = tf.concat([seq_emb, round_emb, lf, det], axis=-1)  # [B, d_model + round_embed_dim + 2]
            tokens.append(token)
        tokens = tf.stack(tokens, axis=1)  # [B, 3, d_model + round_embed_dim + 2]
        tokens = self.token_proj(tokens)  # [B, 3, d_model]
        # No padding in temporal dimension (all 3 rounds always present)
        for layer in self.temporal_layers:
            tokens = layer(tokens, padding_mask=None, training=training)  # [B, 3, d_model]
        tokens = self.final_ln(tokens)  # [B, 3, d_model]
        temporal_pooled = tf.reduce_mean(tokens, axis=1)  # [B, d_model]
        return temporal_pooled


@tf.custom_gradient
def gradient_reversal(x, lam):
    """Gradient reversal operation: forward pass is identity, backward negates gradients.

    Args:
        x: Input tensor.
        lam: Scaling factor for reversed gradients.

    Returns:
        x (unchanged), custom gradient function.
    """
    def grad(dy):
        return -lam * dy, None  # negate gradient, no grad for lam
    return x, grad


class GradientReversalLayer(tf.keras.layers.Layer):
    """Gradient Reversal Layer (Ganin et al. 2016).

    Forward: identity. Backward: negates and scales gradients by lambda.
    Used to force z_bind to NOT encode amplification information.

    Input:  [batch, dim]
    Output: [batch, dim] (same tensor, reversed gradients during backprop)
    """
    def __init__(self, lam=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam

    def call(self, x):
        return gradient_reversal(x, self.lam)


# ==============================================================================
# FULL MODELS
# ==============================================================================

class MLMModel(tf.keras.Model):
    """Masked Language Model for Stage 1 pretraining.

    Trains the sequence encoder via masked token prediction. The encoder
    learns amino acid grammar, positional constraints, and CDR/framework
    structure from sequence data alone (no enrichment info).

    Architecture:
        input_ids [B, L] -> SequenceEncoder -> per_residue [B, L, d_model]
        -> MLM head [B, L, 20] -> cross-entropy at masked positions
    """
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.encoder = SequenceEncoder(
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_encoder_layers,
            ff_dim=args.ff_dim, dropout=args.dropout, max_len=args.max_seq_len,
            pos_encoding=args.pos_encoding, name="seq_encoder",
        )
        self.mlm_head = tf.keras.Sequential([
            tf.keras.layers.Dense(args.d_model, activation="gelu"),  # [B, L, d_model]
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(20),  # [B, L, 20] logits over 20 AAs
        ], name="mlm_head")

    def call(self, inputs, training=False):
        """
        Args:
            inputs: dict with 'input_ids' [B, L], 'padding_mask' [B, L]

        Returns:
            logits: [batch, seq_len, 20] MLM prediction logits
        """
        pooled, per_residue = self.encoder(inputs["input_ids"], inputs["padding_mask"], training=training)
        logits = self.mlm_head(per_residue, training=training)  # [B, L, 20]
        return logits


class MultitaskModel(tf.keras.Model):
    """Multi-task enrichment prediction model for Stage 2.

    Combines sequence encoder, temporal enrichment module, and latent space
    decomposition with four task heads.

    Architecture:
        input_ids [B, L] -> SequenceEncoder -> pooled [B, d_model]
        pooled + log_freqs + detected -> TemporalModule -> temporal [B, d_model]
        combined [B, 2*d_model] -> z_bind [B, Db], z_bio [B, Dbi], z_family [B, Df]

        Heads:
            trajectory:    concat(z_bind, z_bio) -> MLP -> [B, 3] predicted log freqs
            amplification: z_bio -> MLP -> [B, 1] predicted R0 log freq
            grl:           z_bind -> GRL -> MLP -> [B, 1] predicted R0 (adversarial)
            contrastive:   z_family used in NT-Xent loss externally
            mlm_aux:       per_residue -> MLM head -> [B, L, 20] (optional auxiliary)
    """
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        d = args.d_model
        # Sequence encoder (weights loaded from Stage 1)
        self.encoder = SequenceEncoder(
            d_model=d, n_heads=args.n_heads, n_layers=args.n_encoder_layers,
            ff_dim=args.ff_dim, dropout=args.dropout, max_len=args.max_seq_len,
            pos_encoding=args.pos_encoding, name="seq_encoder",
        )
        # Temporal enrichment module
        self.temporal = TemporalEnrichmentModule(
            d_model=d, n_heads=args.n_heads, n_layers=args.n_temporal_layers,
            ff_dim=args.ff_dim, dropout=args.dropout, n_rounds=3,
            round_embed_dim=args.n_round_embed_dim, name="temporal",
        )
        # Latent projection: 2*d_model -> z_bind + z_bio + z_family
        z_total = args.z_bind_dim + args.z_bio_dim + args.z_family_dim
        self.latent_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(d, activation="gelu"),  # [B, 2*d] -> [B, d]
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(z_total),  # [B, d] -> [B, z_total]
        ], name="latent_proj")
        # Task heads
        self.trajectory_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="gelu"),  # [B, Db+Dbi] -> [B, 64]
            tf.keras.layers.Dense(3),  # [B, 64] -> [B, 3] predicted log freqs
        ], name="trajectory_head")
        self.amp_head = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="gelu"),  # [B, Dbi] -> [B, 32]
            tf.keras.layers.Dense(1),  # [B, 32] -> [B, 1] predicted R0
        ], name="amp_head")
        self.grl = GradientReversalLayer(lam=args.grl_lambda, name="grl")
        self.grl_head = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="gelu"),  # [B, Db] -> [B, 32]
            tf.keras.layers.Dense(1),  # [B, 32] -> [B, 1] predicted R0 (adversarial)
        ], name="grl_head")
        # Auxiliary MLM head (optional, same structure as Stage 1)
        self.mlm_head = tf.keras.Sequential([
            tf.keras.layers.Dense(d, activation="gelu"),
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(20),
        ], name="mlm_head_aux")
        # Dimension bookkeeping
        self._z_bind_dim = args.z_bind_dim
        self._z_bio_dim = args.z_bio_dim
        self._z_family_dim = args.z_family_dim

    def encode(self, token_ids, padding_mask, log_freqs, detected, training=False):
        """Full forward pass returning all latent components and predictions.

        Args:
            token_ids:    [B, L] int32
            padding_mask: [B, L] float32
            log_freqs:    [B, 3] float32
            detected:     [B, 3] float32
            training:     bool

        Returns:
            dict with keys:
                'z_bind'      : [B, z_bind_dim]
                'z_bio'       : [B, z_bio_dim]
                'z_family'    : [B, z_family_dim]
                'pred_traj'   : [B, 3] predicted log frequencies
                'pred_amp'    : [B, 1] predicted R0 from z_bio
                'pred_grl'    : [B, 1] predicted R0 from z_bind (reversed grad)
                'per_residue' : [B, L, d_model] for auxiliary MLM
        """
        pooled, per_residue = self.encoder(token_ids, padding_mask, training=training)  # [B, d], [B, L, d]
        temporal = self.temporal(pooled, log_freqs, detected, training=training)  # [B, d]
        combined = tf.concat([pooled, temporal], axis=-1)  # [B, 2*d]
        z_all = self.latent_proj(combined, training=training)  # [B, z_total]
        z_bind = z_all[:, :self._z_bind_dim]  # [B, Db]
        z_bio = z_all[:, self._z_bind_dim:self._z_bind_dim + self._z_bio_dim]  # [B, Dbi]
        z_family = z_all[:, self._z_bind_dim + self._z_bio_dim:]  # [B, Df]
        # Task predictions
        pred_traj = self.trajectory_head(tf.concat([z_bind, z_bio], axis=-1), training=training)  # [B, 3]
        pred_amp = self.amp_head(z_bio, training=training)  # [B, 1]
        z_bind_reversed = self.grl(z_bind)  # [B, Db] (gradients reversed)
        pred_grl = self.grl_head(z_bind_reversed, training=training)  # [B, 1]
        return {
            "z_bind": z_bind, "z_bio": z_bio, "z_family": z_family,
            "pred_traj": pred_traj, "pred_amp": pred_amp, "pred_grl": pred_grl,
            "per_residue": per_residue,
        }

    def call(self, inputs, training=False):
        """Standard forward pass for model building. Use encode() in training loop.

        Args:
            inputs: dict with 'input_ids', 'padding_mask', 'log_freqs', 'detected'

        Returns:
            dict of predictions (same as encode()).
        """
        return self.encode(inputs["input_ids"], inputs["padding_mask"],
                           inputs["log_freqs"], inputs["detected"], training=training)


# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================

def mlm_loss_fn(logits, original_ids, mask_flag):
    """Masked language model cross-entropy loss at masked positions only.

    Args:
        logits:       [B, L, 20] predicted AA logits
        original_ids: [B, L] int32 true token indices (0-19 for AAs)
        mask_flag:    [B, L] float32 (1.0 at masked positions)

    Returns:
        scalar loss (mean CE over masked tokens)
    """
    labels = tf.cast(original_ids, tf.int32)  # [B, L]
    # Clip labels to valid range [0, 19] - pad/mask/unk positions won't matter (masked out)
    labels = tf.clip_by_value(labels, 0, 19)  # [B, L]
    ce = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)  # [B, L]
    masked_ce = ce * mask_flag  # [B, L] zero out non-masked
    n_masked = tf.reduce_sum(mask_flag) + 1e-8
    return tf.reduce_sum(masked_ce) / n_masked  # scalar


def trajectory_loss_fn(pred_traj, true_log_freqs, detected):
    """MSE loss on predicted enrichment trajectories, weighted by detection.

    Sequences detected in a round contribute more to that round's loss.
    Undetected rounds (imputed 0-counts) still contribute but with lower weight.

    Args:
        pred_traj:      [B, 3] predicted standardized log frequencies
        true_log_freqs: [B, 3] target standardized log frequencies
        detected:       [B, 3] detection flags (1=detected, 0=imputed)

    Returns:
        scalar MSE loss
    """
    weights = 0.5 + 0.5 * detected  # [B, 3] range [0.5, 1.0]: detected=1.0, imputed=0.5
    sq_err = tf.square(pred_traj - true_log_freqs)  # [B, 3]
    weighted_se = sq_err * weights  # [B, 3]
    return tf.reduce_mean(weighted_se)  # scalar


def amplification_loss_fn(pred_amp, true_log_freqs):
    """MSE loss for R0 frequency prediction from z_bio.

    Forces z_bio to encode amplification/library composition information
    (R0 is pre-selection, so R0 frequency reflects only biophysics).

    Args:
        pred_amp:       [B, 1] predicted R0 log frequency
        true_log_freqs: [B, 3] target log frequencies (we use column 0)

    Returns:
        scalar MSE loss
    """
    true_r0 = true_log_freqs[:, 0:1]  # [B, 1]
    return tf.reduce_mean(tf.square(pred_amp - true_r0))  # scalar


def grl_loss_fn(pred_grl, true_log_freqs):
    """MSE loss for gradient-reversed R0 prediction from z_bind.

    The GRL negates gradients, so minimizing this loss on the forward pass
    actually pushes z_bind AWAY from encoding R0 information. This enforces
    disentanglement: z_bind should NOT predict pre-selection frequency.

    Args:
        pred_grl:       [B, 1] predicted R0 from z_bind (gradient reversed)
        true_log_freqs: [B, 3] target log frequencies (column 0 = R0)

    Returns:
        scalar MSE loss
    """
    true_r0 = true_log_freqs[:, 0:1]  # [B, 1]
    return tf.reduce_mean(tf.square(pred_grl - true_r0))  # scalar


def nt_xent_loss_fn(z_orig, z_aug, temperature):
    """Normalized Temperature-scaled Cross-Entropy (NT-Xent) contrastive loss.

    Within a batch of B sequences, each sequence has an original z and an
    augmented z (from random AA mutations). Positive pairs are (z_i, z_i_aug).
    Negative pairs are all other 2B-2 samples.

    Args:
        z_orig:      [B, D] embeddings of original sequences
        z_aug:       [B, D] embeddings of augmented (mutated) sequences
        temperature: float, temperature scaling factor

    Returns:
        scalar NT-Xent loss
    """
    B = tf.shape(z_orig)[0]
    # L2 normalize
    z_orig_n = tf.math.l2_normalize(z_orig, axis=-1)  # [B, D]
    z_aug_n = tf.math.l2_normalize(z_aug, axis=-1)     # [B, D]
    # Concatenate: [z_orig; z_aug] -> [2B, D]
    z_all = tf.concat([z_orig_n, z_aug_n], axis=0)  # [2B, D]
    # Cosine similarity matrix: [2B, 2B]
    sim_matrix = tf.matmul(z_all, z_all, transpose_b=True) / temperature  # [2B, 2B]
    # Mask out self-similarity (diagonal)
    mask_self = tf.eye(2 * B, dtype=tf.float32) * -1e9  # [2B, 2B]
    sim_matrix = sim_matrix + mask_self  # [2B, 2B]
    # Positive pairs: (i, i+B) and (i+B, i)
    # For row i (i < B), positive is at column i+B
    # For row i+B, positive is at column i
    labels_top = tf.range(B, 2 * B)  # [B] indices B..2B-1
    labels_bot = tf.range(0, B)      # [B] indices 0..B-1
    labels = tf.concat([labels_top, labels_bot], axis=0)  # [2B]
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, sim_matrix, from_logits=True)  # [2B]
    return tf.reduce_mean(loss)  # scalar


# ==============================================================================
# OPTIMIZER AND SCHEDULE
# ==============================================================================

def build_optimizer(args, total_steps):
    """Build optimizer with optional learning rate schedule.

    Args:
        args: Parsed arguments with optimizer, learning_rate, lr_schedule, etc.
        total_steps (int): Total training steps for schedule computation.

    Returns:
        tf.keras.optimizers.Optimizer
    """
    # Build learning rate schedule
    if args.lr_schedule == "constant":
        lr = args.learning_rate
    elif args.lr_schedule == "cosine":
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.learning_rate, decay_steps=total_steps, alpha=1e-6)
    elif args.lr_schedule == "warmup_cosine":
        # Linear warmup then cosine decay
        warmup_lr = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=1e-7, decay_steps=args.warmup_steps,
            end_learning_rate=args.learning_rate, power=1.0)
        cosine_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=max(1, total_steps - args.warmup_steps), alpha=1e-6)
        # Use a custom schedule that switches at warmup_steps
        class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, warmup_sched, cosine_sched, warmup_steps):
                super().__init__()
                self.warmup_sched = warmup_sched
                self.cosine_sched = cosine_sched
                self.warmup_steps = warmup_steps
            def __call__(self, step):
                return tf.cond(step < self.warmup_steps,
                               lambda: self.warmup_sched(step),
                               lambda: self.cosine_sched(step - self.warmup_steps))
            def get_config(self):
                return {"warmup_steps": self.warmup_steps}
        lr = WarmupCosine(warmup_lr, cosine_lr, args.warmup_steps)
    else:
        lr = args.learning_rate
    # Build optimizer
    if args.optimizer == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif args.optimizer == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif args.optimizer == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        return tf.keras.optimizers.Adam(learning_rate=lr)


# ==============================================================================
# TRAINING LOOPS
# ==============================================================================

def train_mlm(args, data):
    """Stage 1: Masked Language Model pretraining of the sequence encoder.

    Trains the SequenceEncoder + MLM head using masked token prediction
    on all sequences (enrichment information is ignored).

    Args:
        args: Parsed arguments.
        data: dict from load_and_merge_rounds().
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: MLM PRETRAINING")
    logger.info("=" * 60)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    n_total = len(data["ids"])
    # Data splitting
    if args.split:
        train_idx, val_idx, test_idx = split_indices(n_total, args.val_frac, args.test_frac, args.seed)
        logger.info("Split: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))
    else:
        train_idx = np.arange(n_total)
        val_idx = np.arange(min(n_total, 10000))  # small val for monitoring
        logger.info("No split: using all %d for training", n_total)
    # Build datasets
    train_ds, n_train = build_mlm_dataset(data["seq_indices"], data["seq_lengths"], train_idx,
                                           args.batch_size, args.mlm_mask_frac, args.seed, shuffle=True)
    val_ds, n_val = build_mlm_dataset(data["seq_indices"], data["seq_lengths"], val_idx,
                                       args.batch_size, args.mlm_mask_frac, args.seed + 1, shuffle=False)
    steps_per_epoch = n_train // args.batch_size
    val_steps = max(1, n_val // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    logger.info("Steps/epoch: %d, Total steps: %d", steps_per_epoch, total_steps)
    # Build model
    model = MLMModel(args)
    # Build model by running a dummy forward pass
    dummy = {"input_ids": tf.zeros([1, args.max_seq_len], dtype=tf.int32),
             "padding_mask": tf.ones([1, args.max_seq_len], dtype=tf.float32)}
    _ = model(dummy, training=False)
    model.summary(print_fn=logger.info)
    # Optimizer
    optimizer = build_optimizer(args, total_steps)
    # Training
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss_accum, train_steps = 0.0, 0
        for batch in train_ds.take(steps_per_epoch):
            with tf.GradientTape() as tape:
                logits = model({"input_ids": batch["input_ids"], "padding_mask": batch["padding_mask"]}, training=True)  # [B, L, 20]
                loss = mlm_loss_fn(logits, batch["original_ids"], batch["mask_flag"])
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss_accum += float(loss)
            train_steps += 1
        train_loss = train_loss_accum / max(train_steps, 1)
        # Validation
        val_loss_accum, val_steps_done = 0.0, 0
        for batch in val_ds.take(val_steps):
            logits = model({"input_ids": batch["input_ids"], "padding_mask": batch["padding_mask"]}, training=False)
            loss = mlm_loss_fn(logits, batch["original_ids"], batch["mask_flag"])
            val_loss_accum += float(loss)
            val_steps_done += 1
        val_loss = val_loss_accum / max(val_steps_done, 1)
        elapsed = time.time() - t0
        logger.info("Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  time=%.1fs", epoch, args.epochs, train_loss, val_loss, elapsed)
        # Checkpoint best
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(args.checkpoint_dir, "mlm_best.weights.h5")
            model.save_weights(save_path)
            logger.info("  -> Saved best MLM weights (val_loss=%.4f)", val_loss)
        else:
            patience_counter += 1
        # Convergence check
        if args.convergence and patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d (patience=%d exhausted)", epoch, args.patience)
            break
    # Save final weights
    final_path = os.path.join(args.checkpoint_dir, "mlm_final.weights.h5")
    model.save_weights(final_path)
    logger.info("MLM pretraining complete. Best val_loss=%.4f", best_val_loss)
    return model


def train_multitask(args, data):
    """Stage 2: Multi-task enrichment model training.

    Loads pretrained MLM encoder weights, then trains the full pipeline
    with trajectory prediction, amplification, GRL disentanglement,
    contrastive clustering, and optional auxiliary MLM losses.

    Args:
        args: Parsed arguments (requires args.mlm_weights).
        data: dict from load_and_merge_rounds().
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: MULTI-TASK ENRICHMENT TRAINING")
    logger.info("=" * 60)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    n_total = len(data["ids"])
    # Data splitting
    if args.split:
        train_idx, val_idx, test_idx = split_indices(n_total, args.val_frac, args.test_frac, args.seed)
        logger.info("Split: train=%d, val=%d, test=%d", len(train_idx), len(val_idx), len(test_idx))
    else:
        train_idx = np.arange(n_total)
        val_idx = np.arange(min(n_total, 10000))
        logger.info("No split: using all %d for training", n_total)
    # Build datasets
    train_ds, n_train = build_multitask_dataset(
        data["seq_indices"], data["seq_lengths"], data["log_freqs"], data["detected"],
        train_idx, args.batch_size, args.seed, shuffle=True,
        mask_frac=args.mlm_mask_frac, n_mutations=args.n_mutations)
    val_ds, n_val = build_multitask_dataset(
        data["seq_indices"], data["seq_lengths"], data["log_freqs"], data["detected"],
        val_idx, args.batch_size, args.seed + 1, shuffle=False,
        mask_frac=args.mlm_mask_frac, n_mutations=args.n_mutations)
    steps_per_epoch = n_train // args.batch_size
    val_steps = max(1, n_val // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    logger.info("Steps/epoch: %d, Total steps: %d", steps_per_epoch, total_steps)
    # Build model
    model = MultitaskModel(args)
    # Build model with dummy forward pass
    dummy = {
        "input_ids": tf.zeros([1, args.max_seq_len], dtype=tf.int32),
        "padding_mask": tf.ones([1, args.max_seq_len], dtype=tf.float32),
        "log_freqs": tf.zeros([1, 3], dtype=tf.float32),
        "detected": tf.ones([1, 3], dtype=tf.float32),
    }
    _ = model(dummy, training=False)
    model.summary(print_fn=logger.info)
    # Load pretrained MLM encoder weights
    if args.mlm_weights:
        logger.info("Loading MLM encoder weights from: %s", args.mlm_weights)
        mlm_model = MLMModel(args)
        _ = mlm_model({"input_ids": tf.zeros([1, args.max_seq_len], dtype=tf.int32),
                        "padding_mask": tf.ones([1, args.max_seq_len], dtype=tf.float32)}, training=False)
        mlm_model.load_weights(args.mlm_weights)
        # Transfer encoder weights
        for mt_var, mlm_var in zip(model.encoder.trainable_variables, mlm_model.encoder.trainable_variables):
            mt_var.assign(mlm_var)
        logger.info("  -> Encoder weights transferred (%d variables)", len(model.encoder.trainable_variables))
        del mlm_model
    else:
        logger.warning("No MLM weights provided. Training encoder from scratch in multitask stage.")
    # Optimizer
    optimizer = build_optimizer(args, total_steps)
    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses_accum = defaultdict(float)
        train_steps = 0
        for batch in train_ds.take(steps_per_epoch):
            with tf.GradientTape() as tape:
                # Forward pass: original sequences
                out = model.encode(batch["input_ids"], batch["padding_mask"],
                                   batch["log_freqs"], batch["detected"], training=True)
                # --- Trajectory loss ---
                l_traj = trajectory_loss_fn(out["pred_traj"], batch["log_freqs"], batch["detected"])
                # --- Amplification loss (z_bio -> R0) ---
                l_amp = amplification_loss_fn(out["pred_amp"], batch["log_freqs"])
                # --- Gradient reversal loss (z_bind -/-> R0) ---
                l_grl = grl_loss_fn(out["pred_grl"], batch["log_freqs"])
                # --- Contrastive loss (z_family) ---
                # Forward pass on augmented sequences (share same encoder)
                out_aug = model.encode(batch["aug_input_ids"], batch["padding_mask"],
                                       batch["log_freqs"], batch["detected"], training=True)
                l_contrast = nt_xent_loss_fn(out["z_family"], out_aug["z_family"], args.contrastive_temp)
                # --- Auxiliary MLM loss (optional) ---
                l_mlm = tf.constant(0.0)
                if args.w_mlm_aux > 0:
                    mlm_logits = model.mlm_head(out["per_residue"], training=True)  # [B, L, 20]
                    # Use original input_ids (not mlm_input_ids) as labels, mlm_mask_flag for positions
                    l_mlm = mlm_loss_fn(mlm_logits, batch["input_ids"], batch["mlm_mask_flag"])
                # --- Total loss ---
                total_loss = (args.w_trajectory * l_traj + args.w_amplification * l_amp +
                              args.w_grl * l_grl + args.w_contrastive * l_contrast +
                              args.w_mlm_aux * l_mlm)
            grads = tape.gradient(total_loss, model.trainable_variables)
            # Gradient clipping for stability
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            losses_accum["total"] += float(total_loss)
            losses_accum["traj"] += float(l_traj)
            losses_accum["amp"] += float(l_amp)
            losses_accum["grl"] += float(l_grl)
            losses_accum["contrast"] += float(l_contrast)
            losses_accum["mlm"] += float(l_mlm)
            train_steps += 1
        # Average training losses
        for k in losses_accum:
            losses_accum[k] /= max(train_steps, 1)
        # Validation
        val_losses_accum = defaultdict(float)
        val_steps_done = 0
        for batch in val_ds.take(val_steps):
            out = model.encode(batch["input_ids"], batch["padding_mask"],
                               batch["log_freqs"], batch["detected"], training=False)
            l_traj = trajectory_loss_fn(out["pred_traj"], batch["log_freqs"], batch["detected"])
            l_amp = amplification_loss_fn(out["pred_amp"], batch["log_freqs"])
            l_grl = grl_loss_fn(out["pred_grl"], batch["log_freqs"])
            out_aug = model.encode(batch["aug_input_ids"], batch["padding_mask"],
                                   batch["log_freqs"], batch["detected"], training=False)
            l_contrast = nt_xent_loss_fn(out["z_family"], out_aug["z_family"], args.contrastive_temp)
            total_val = (args.w_trajectory * l_traj + args.w_amplification * l_amp +
                         args.w_grl * l_grl + args.w_contrastive * l_contrast)
            val_losses_accum["total"] += float(total_val)
            val_losses_accum["traj"] += float(l_traj)
            val_losses_accum["amp"] += float(l_amp)
            val_losses_accum["grl"] += float(l_grl)
            val_losses_accum["contrast"] += float(l_contrast)
            val_steps_done += 1
        for k in val_losses_accum:
            val_losses_accum[k] /= max(val_steps_done, 1)
        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d [%.1fs]  TRAIN total=%.4f traj=%.4f amp=%.4f grl=%.4f ctr=%.4f mlm=%.4f  |  VAL total=%.4f traj=%.4f",
            epoch, args.epochs, elapsed,
            losses_accum["total"], losses_accum["traj"], losses_accum["amp"],
            losses_accum["grl"], losses_accum["contrast"], losses_accum["mlm"],
            val_losses_accum["total"], val_losses_accum["traj"],
        )
        # Checkpoint
        val_loss = val_losses_accum["total"]
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(args.checkpoint_dir, "multitask_best.weights.h5")
            model.save_weights(save_path)
            logger.info("  -> Saved best multitask weights (val_loss=%.4f)", val_loss)
        else:
            patience_counter += 1
        if args.convergence and patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
            break
    final_path = os.path.join(args.checkpoint_dir, "multitask_final.weights.h5")
    model.save_weights(final_path)
    logger.info("Multitask training complete. Best val_loss=%.4f", best_val_loss)
    # Save normalization stats for inference
    np.savez(os.path.join(args.checkpoint_dir, "norm_stats.npz"),
             log_freq_mean=data["log_freq_mean"], log_freq_std=data["log_freq_std"],
             total_counts=data["total_counts"])
    logger.info("Saved normalization stats to %s/norm_stats.npz", args.checkpoint_dir)
    return model


# ==============================================================================
# INFERENCE
# ==============================================================================

def run_inference(args, data):
    """Run inference with a trained multitask model.

    Produces per-sequence predictions: enrichment trajectory, latent embeddings
    (z_bind, z_bio, z_family), amplification score, and binding rank.

    Args:
        args: Parsed arguments (requires args.multitask_weights).
        data: dict from load_and_merge_rounds().

    Outputs:
        Saves TSV file with columns:
            seq_id, z_bind_score, z_bio_score, pred_log_freq_R0, pred_log_freq_R1,
            pred_log_freq_R3, pred_amp_R0, actual_log_freq_R0, actual_log_freq_R1,
            actual_log_freq_R3, detected_R0, detected_R1, detected_R3

        Saves NPZ file with full latent embeddings:
            z_bind [N, z_bind_dim], z_bio [N, z_bio_dim], z_family [N, z_family_dim]
    """
    logger.info("=" * 60)
    logger.info("INFERENCE")
    logger.info("=" * 60)
    if not args.multitask_weights:
        raise ValueError("--multitask_weights is required for inference stage.")
    # Build and load model
    model = MultitaskModel(args)
    dummy = {
        "input_ids": tf.zeros([1, args.max_seq_len], dtype=tf.int32),
        "padding_mask": tf.ones([1, args.max_seq_len], dtype=tf.float32),
        "log_freqs": tf.zeros([1, 3], dtype=tf.float32),
        "detected": tf.ones([1, 3], dtype=tf.float32),
    }
    _ = model(dummy, training=False)
    model.load_weights(args.multitask_weights)
    logger.info("Loaded multitask weights from: %s", args.multitask_weights)
    # Load normalization stats
    norm_path = os.path.join(os.path.dirname(args.multitask_weights), "norm_stats.npz")
    if os.path.exists(norm_path):
        norm_stats = np.load(norm_path)
        lf_mean, lf_std = norm_stats["log_freq_mean"], norm_stats["log_freq_std"]
        logger.info("Loaded normalization stats from: %s", norm_path)
    else:
        lf_mean, lf_std = data["log_freq_mean"], data["log_freq_std"]
        logger.warning("Using normalization stats from current data (norm_stats.npz not found at %s)", norm_path)
    n_total = len(data["ids"])
    batch_size = args.inference_batch_size
    max_len = args.max_seq_len
    # Preallocate output arrays
    all_z_bind = np.zeros((n_total, args.z_bind_dim), dtype=np.float32)
    all_z_bio = np.zeros((n_total, args.z_bio_dim), dtype=np.float32)
    all_z_family = np.zeros((n_total, args.z_family_dim), dtype=np.float32)
    all_pred_traj = np.zeros((n_total, 3), dtype=np.float32)
    all_pred_amp = np.zeros((n_total, 1), dtype=np.float32)
    # Batch inference
    n_batches = (n_total + batch_size - 1) // batch_size
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_total)
        actual_bs = end - start
        seq_batch = data["seq_indices"][start:end].astype(np.int32)  # [bs, L]
        lf_batch = data["log_freqs"][start:end]                      # [bs, 3]
        det_batch = data["detected"][start:end]                      # [bs, 3]
        # Padding mask
        lens_batch = data["seq_lengths"][start:end]  # [bs]
        pad_mask = np.zeros((actual_bs, max_len), dtype=np.float32)
        for i in range(actual_bs):
            pad_mask[i, :lens_batch[i]] = 1.0
        # Forward pass
        out = model.encode(
            tf.constant(seq_batch), tf.constant(pad_mask),
            tf.constant(lf_batch), tf.constant(det_batch), training=False)
        all_z_bind[start:end] = out["z_bind"].numpy()
        all_z_bio[start:end] = out["z_bio"].numpy()
        all_z_family[start:end] = out["z_family"].numpy()
        all_pred_traj[start:end] = out["pred_traj"].numpy()
        all_pred_amp[start:end] = out["pred_amp"].numpy()
        if (b + 1) % 100 == 0 or b == n_batches - 1:
            logger.info("  Inference batch %d/%d", b + 1, n_batches)
    # Compute summary scores
    z_bind_score = np.linalg.norm(all_z_bind, axis=1)  # [N] L2 norm as binding strength proxy
    z_bio_score = np.linalg.norm(all_z_bio, axis=1)    # [N] biophysical score
    # De-standardize predictions for interpretability
    pred_traj_raw = all_pred_traj * lf_std + lf_mean  # [N, 3] raw log2 frequencies
    pred_amp_raw = all_pred_amp * lf_std[0, 0] + lf_mean[0, 0]  # [N, 1]
    actual_lf_raw = data["log_freqs_raw"]  # [N, 3]
    # Save outputs
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # TSV predictions
    tsv_path = args.output_prefix + "_predictions.tsv"
    with open(tsv_path, "w") as f:
        header = "\t".join([
            "seq_id", "z_bind_score", "z_bio_score",
            "pred_log_freq_R0", "pred_log_freq_R1", "pred_log_freq_R3", "pred_amp_R0",
            "actual_log_freq_R0", "actual_log_freq_R1", "actual_log_freq_R3",
            "detected_R0", "detected_R1", "detected_R3",
        ])
        f.write(header + "\n")
        for i in range(n_total):
            row = [
                data["ids"][i],
                f"{z_bind_score[i]:.6f}", f"{z_bio_score[i]:.6f}",
                f"{pred_traj_raw[i, 0]:.6f}", f"{pred_traj_raw[i, 1]:.6f}", f"{pred_traj_raw[i, 2]:.6f}",
                f"{pred_amp_raw[i, 0]:.6f}",
                f"{actual_lf_raw[i, 0]:.6f}", f"{actual_lf_raw[i, 1]:.6f}", f"{actual_lf_raw[i, 2]:.6f}",
                f"{data['detected'][i, 0]:.0f}", f"{data['detected'][i, 1]:.0f}", f"{data['detected'][i, 2]:.0f}",
            ]
            f.write("\t".join(row) + "\n")
    logger.info("Saved predictions: %s", tsv_path)
    # NPZ latent embeddings (for downstream clustering, visualization, generation)
    npz_path = args.output_prefix + "_latents.npz"
    np.savez_compressed(npz_path,
                        ids=np.array(data["ids"], dtype=object),
                        z_bind=all_z_bind, z_bio=all_z_bio, z_family=all_z_family,
                        pred_traj=all_pred_traj, pred_amp=all_pred_amp,
                        z_bind_score=z_bind_score, z_bio_score=z_bio_score)
    logger.info("Saved latent embeddings: %s", npz_path)
    logger.info("Inference complete for %d sequences.", n_total)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main():
    args = parse_args()
    set_seeds(args.seed)
    # Mixed precision
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision enabled (float16).")
    # Device info
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("GPU(s) detected: %s", [g.name for g in gpus])
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.info("No GPU detected. Running on CPU.")
    # Load data
    logger.info("Loading and merging round data...")
    data = load_and_merge_rounds(args.r0, args.r1, args.r3, args.max_seq_len, args.pseudocount)
    # Dispatch to stage
    if args.stage == "mlm":
        train_mlm(args, data)
    elif args.stage == "multitask":
        train_multitask(args, data)
    elif args.stage == "inference":
        run_inference(args, data)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()