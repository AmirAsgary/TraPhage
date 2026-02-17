#!/usr/bin/env python3
"""
================================================================================
Nanobody Phage Display Pipeline — MLM + Autoregressive VAE (v3)
================================================================================
End-to-end pipeline:
  Stage 1 — MLM: Masked language model pretraining on R0 sequences.
  Stage 2 — Autoregressive VAE: Random-order autoregressive decoding of
            round presence/frequency with partial masking, followed by
            VAE reconstruction of sequences and round labels.

Usage examples:
  # Preprocess + MLM
  python pipeline.py --r0_train r0.fa --r1_train r1.fa --r3_train r3.fa \
      --output_dir out --mlm --epochs 20

  # Autoregressive VAE (loads MLM checkpoint)
  python pipeline.py --r0_train r0.fa --r1_train r1.fa --r3_train r3.fa \
      --output_dir out --autoregressive_vae --epochs 50

  # Inference with round masking
  python pipeline.py --r0_train r0.fa --r1_train r1.fa --r3_train r3.fa \
      --output_dir out --inference --inference_mask_rounds R1,R3

  # Generator: sample from latent space
  python pipeline.py --r0_train r0.fa --r1_train r1.fa --r3_train r3.fa \
      --output_dir out --generator --n_generate 1000

  # Analysis (AUC, PR, UMAP, training curves)
  python pipeline.py --r0_train r0.fa --r1_train r1.fa --r3_train r3.fa \
      --output_dir out --analysis
================================================================================
"""

import os, sys, glob, math, time, json, random, logging, argparse
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================
AA = "ACDEFGHIKLMNPQRSTVWY"
AA2I = {a: i for i, a in enumerate(AA)}
PAD, MASK, UNK = 20, 21, 22
VOCAB = 23
N_AA = 20
ROUND_NAMES = ["R0", "R1", "R3"]

BLOSUM62 = np.array([
    [ 4, 0,-2,-1,-2, 0,-2,-1,-1,-1,-1,-2,-1,-1,-1, 1, 0, 0,-3,-2],
    [ 0, 9,-3,-4,-2,-3,-3,-1,-3,-1,-1,-3,-3,-3,-3,-1,-1,-1,-2,-2],
    [-2,-3, 6, 2,-3,-1,-1,-3,-1,-4,-3, 1,-1, 0,-2, 0,-1,-3,-4,-3],
    [-1,-4, 2, 5,-3,-2, 0,-3, 1,-3,-2, 0,-1, 2, 0, 0,-1,-2,-3,-2],
    [-2,-2,-3,-3, 6,-3,-1, 0,-3, 0, 0,-3,-4,-3,-3,-2,-2,-1, 1, 3],
    [ 0,-3,-1,-2,-3, 6,-2,-4,-2,-4,-3, 0,-2,-2,-2, 0,-2,-3,-2,-3],
    [-2,-3,-1, 0,-1,-2, 8,-3,-1,-3,-2, 1,-2, 0, 0,-1,-2,-3,-2, 2],
    [-1,-1,-3,-3, 0,-4,-3, 4,-3, 2, 1,-3,-3,-3,-3,-2,-1, 3,-3,-1],
    [-1,-3,-1, 1,-3,-2,-1,-3, 5,-2,-1, 0,-1, 1, 2, 0,-1,-2,-3,-2],
    [-1,-1,-4,-3, 0,-4,-3, 2,-2, 4, 2,-3,-3,-2,-2,-2,-1, 1,-2,-1],
    [-1,-1,-3,-2, 0,-3,-2, 1,-1, 2, 5,-2,-2, 0,-1,-1,-1, 1,-1,-1],
    [-2,-3, 1, 0,-3, 0, 1,-3, 0,-3,-2, 6,-2, 0, 0, 1, 0,-3,-4,-2],
    [-1,-3,-1,-1,-4,-2,-2,-3,-1,-3,-2,-2, 7,-1,-2,-1,-1,-2,-4,-3],
    [-1,-3, 0, 2,-3,-2, 0,-3, 1,-2, 0, 0,-1, 5, 1, 0,-1,-2,-2,-1],
    [-1,-3,-2, 0,-3,-2, 0,-3, 2,-2,-1, 0,-2, 1, 5,-1,-1,-3,-3,-2],
    [ 1,-1, 0, 0,-2, 0,-1,-2, 0,-2,-1, 1,-1, 0,-1, 4, 1,-2,-3,-2],
    [ 0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1, 0,-1,-1,-1, 1, 5, 0,-2,-2],
    [ 0,-1,-3,-2,-1,-3,-3, 3,-2, 1, 1,-3,-2,-2,-3,-2, 0, 4,-3,-1],
    [-3,-2,-4,-3, 1,-2,-2,-3,-3,-2,-1,-4,-4,-2,-3,-3,-2,-3,11, 2],
    [-2,-2,-3,-2, 3,-3, 2,-1,-2,-1,-1,-2,-3,-1,-2,-2,-2,-1, 2, 7],
], dtype=np.float32)
_bm = BLOSUM62.mean(1, keepdims=True)
_bs = BLOSUM62.std(1, keepdims=True) + 1e-8
BLOSUM62_NORM = (BLOSUM62 - _bm) / _bs
BLOSUM62_FULL = np.vstack([BLOSUM62_NORM, np.zeros((3, 20), dtype=np.float32)])  # [23,20]

ALL_PERMS = tf.constant([
    [0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]
], dtype=tf.int32)  # all 6 permutations of 3 rounds


# =============================================================================
# ARGUMENT PARSING
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Nanobody Pipeline v3: MLM + Autoregressive VAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--r0_train", required=True, help="R0 train FASTA.")
    g.add_argument("--r1_train", required=True, help="R1 train FASTA.")
    g.add_argument("--r3_train", required=True, help="R3 train FASTA.")
    g.add_argument("--r0_val", default=None, help="R0 val FASTA (optional).")
    g.add_argument("--r1_val", default=None, help="R1 val FASTA (optional).")
    g.add_argument("--r3_val", default=None, help="R3 val FASTA (optional).")
    g.add_argument("--val_split", type=float, default=0.1, help="Val fraction if no val FASTAs.")
    g.add_argument("--test_split", type=float, default=0.05, help="Test fraction if no val FASTAs.")
    g.add_argument("--max_len", type=int, default=141, help="Pad/truncate length.")
    g.add_argument("--pseudocount", type=float, default=0.5)
    g.add_argument("--tfrecord_shards", type=int, default=32)
    g.add_argument("--force_tfrecord", action="store_true")
    # Output
    p.add_argument("--output_dir", required=True, help="Root output directory.")
    # Stages
    g2 = p.add_argument_group("Pipeline stages")
    g2.add_argument("--mlm", action="store_true", help="Train MLM on R0.")
    g2.add_argument("--autoregressive_vae", action="store_true", help="Train autoreg + VAE.")
    g2.add_argument("--inference", action="store_true", help="Run inference.")
    g2.add_argument("--generator", action="store_true", help="Generate from VAE decoder.")
    g2.add_argument("--analysis", action="store_true", help="AUC/PR/UMAP analysis.")
    # MLM
    g3 = p.add_argument_group("MLM architecture")
    g3.add_argument("--d_model", type=int, default=32)
    g3.add_argument("--n_heads", type=int, default=2)
    g3.add_argument("--n_layers", type=int, default=1)
    g3.add_argument("--ff_dim", type=int, default=64)
    g3.add_argument("--dropout", type=float, default=0.1)
    g3.add_argument("--l1_reg", type=float, default=0.0)
    g3.add_argument("--l2_reg", type=float, default=0.0)
    g3.add_argument("--mlm_mask_frac", type=float, default=0.15)
    g3.add_argument("--pos_encoding", default="rope", choices=["rope","learned","sinusoidal"])
    # Autoregressive VAE
    g4 = p.add_argument_group("Autoregressive VAE")
    g4.add_argument("--autoreg_hidden_dim", type=int, default=64)
    g4.add_argument("--autoreg_mask_rate", type=float, default=0.5)
    g4.add_argument("--vae_hidden_dim", type=int, default=64)
    g4.add_argument("--vae_latent_dim", type=int, default=64)
    g4.add_argument("--kl_weight", type=float, default=1e-3, help="KL divergence weight.")
    g4.add_argument("--w_recon_seq", type=float, default=1.0)
    g4.add_argument("--w_recon_round", type=float, default=1.0)
    g4.add_argument("--w_autoreg", type=float, default=1.0)
    # Training
    g5 = p.add_argument_group("Training")
    g5.add_argument("--epochs", type=int, default=50)
    g5.add_argument("--batch_size", type=int, default=256)
    g5.add_argument("--learning_rate", type=float, default=1e-3)
    g5.add_argument("--optimizer", default="adam", choices=["adam","adamw","sgd"])
    g5.add_argument("--weight_decay", type=float, default=0.0)
    g5.add_argument("--lr_schedule", default="warmup_cosine", choices=["constant","cosine","warmup_cosine"])
    g5.add_argument("--warmup_steps", type=int, default=500)
    g5.add_argument("--grad_clip", type=float, default=1.0)
    g5.add_argument("--convergence", action="store_true", help="Early stopping.")
    g5.add_argument("--patience", type=int, default=5)
    g5.add_argument("--min_delta", type=float, default=1e-4)
    g5.add_argument("--log_every", type=int, default=100)
    g5.add_argument("--ckpt_every", type=int, default=1000)
    g5.add_argument("--resume", action="store_true")
    # Freezing
    g6 = p.add_argument_group("Freezing")
    g6.add_argument("--freeze_mlm", action="store_true")
    g6.add_argument("--freeze_autoreg", action="store_true")
    g6.add_argument("--freeze_vae", action="store_true")
    # Inference / generator
    g7 = p.add_argument_group("Inference / Generator")
    g7.add_argument("--inference_batch_size", type=int, default=512)
    g7.add_argument("--inference_mask_rounds", type=str, default="",
                    help="Comma-separated rounds to mask during inference, e.g. 'R1,R3'.")
    g7.add_argument("--n_generate", type=int, default=100, help="Sequences to generate.")
    # Performance
    g8 = p.add_argument_group("Performance")
    g8.add_argument("--mixed_precision", action="store_true")
    g8.add_argument("--xla", action="store_true")
    g8.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seeds(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)


def get_reg(args):
    """Build kernel regularizer from l1/l2 args."""
    if args.l1_reg > 0 and args.l2_reg > 0:
        return tf.keras.regularizers.L1L2(args.l1_reg, args.l2_reg)
    if args.l1_reg > 0:
        return tf.keras.regularizers.L1(args.l1_reg)
    if args.l2_reg > 0:
        return tf.keras.regularizers.L2(args.l2_reg)
    return None


# =============================================================================
# DATA LOADING & TFRECORD I/O
# =============================================================================
def parse_fasta(path):
    """Yield (id, count, seq) from FASTA with >id_count headers."""
    sid, cnt, parts = None, None, []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith(">"):
                if sid is not None:
                    yield sid, cnt, "".join(parts)
                toks = ln[1:].rsplit("_", 1)
                sid, cnt, parts = toks[0], float(toks[1]), []
            else:
                parts.append(ln.upper())
    if sid is not None:
        yield sid, cnt, "".join(parts)


def load_round_files(r0p, r1p, r3p, max_len, pseudocount=0.5):
    """Parse three FASTA round files, merge by ID, compute freq & presence."""
    log.info("Loading R0: %s", r0p)
    r0 = {s: (c, q) for s, c, q in parse_fasta(r0p)}
    log.info("  %d seqs", len(r0))
    log.info("Loading R1: %s", r1p)
    r1 = {s: c for s, c, _ in parse_fasta(r1p)}
    log.info("  %d seqs", len(r1))
    log.info("Loading R3: %s", r3p)
    r3 = {s: c for s, c, _ in parse_fasta(r3p)}
    log.info("  %d seqs", len(r3))

    all_ids = sorted(r0.keys())
    n = len(all_ids)
    tot = np.array([sum(c for c, _ in r0.values()),
                    sum(r1.values()) or 1.0,
                    sum(r3.values()) or 1.0], dtype=np.float64)
    log.info("Totals: R0=%.0f R1=%.0f R3=%.0f", *tot)

    seqs = np.full((n, max_len), PAD, dtype=np.int8)
    slens = np.zeros(n, dtype=np.int16)
    counts = np.zeros((n, 3), dtype=np.float32)
    for i, sid in enumerate(all_ids):
        c0, sq = r0[sid]
        L = min(len(sq), max_len)
        slens[i] = L
        for j in range(L):
            seqs[i, j] = AA2I.get(sq[j], UNK)
        counts[i] = [c0, r1.get(sid, 0.0), r3.get(sid, 0.0)]
        if (i + 1) % 2_000_000 == 0:
            log.info("  encoded %d/%d", i + 1, n)

    freq = (counts / tot[None, :].astype(np.float32)).astype(np.float32)
    pres = (counts > 0).astype(np.float32)
    # For R0 sequences with pseudocount detection
    pres[:, 0] = np.where(counts[:, 0] > pseudocount, 1.0, 0.0)
    log.info("Encoded %d sequences", n)
    return {"seqs": seqs, "slens": slens, "counts": counts, "freq": freq,
            "pres": pres, "ids": all_ids, "totals": tot}


def _bf(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

def _if(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def write_tfrecords(data, indices, out_dir, prefix, n_shards, max_len):
    os.makedirs(out_dir, exist_ok=True)
    n = len(indices)
    per = (n + n_shards - 1) // n_shards
    written = 0
    for sh in range(n_shards):
        s, e = sh * per, min((sh + 1) * per, n)
        if s >= n:
            break
        path = os.path.join(out_dir, f"{prefix}_{sh:05d}.tfrecord")
        with tf.io.TFRecordWriter(path) as w:
            for idx in indices[s:e]:
                feat = {
                    "seq": _bf(data["seqs"][idx].tobytes()),
                    "slen": _if(int(data["slens"][idx])),
                    "freq": _bf(data["freq"][idx].tobytes()),
                    "pres": _bf(data["pres"][idx].tobytes()),
                }
                w.write(tf.train.Example(
                    features=tf.train.Features(feature=feat)).SerializeToString())
                written += 1
    log.info("  %s: %d records in %d shards", prefix, written, n_shards)
    return written


def preprocess_data(args):
    """Full FASTA → TFRecord pipeline. Returns metadata dict."""
    tfdir = os.path.join(args.output_dir, "tfrecords")
    data = load_round_files(args.r0_train, args.r1_train, args.r3_train,
                            args.max_len, args.pseudocount)
    n = len(data["ids"])

    has_val_files = (args.r0_val and args.r1_val and args.r3_val)
    if has_val_files:
        vdata = load_round_files(args.r0_val, args.r1_val, args.r3_val,
                                 args.max_len, args.pseudocount)
        train_idx = np.arange(n)
        val_idx_src = "external"
    else:
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n)
        nt = int(n * args.test_split)
        nv = int(n * args.val_split)
        test_idx, val_idx, train_idx = perm[:nt], perm[nt:nt+nv], perm[nt+nv:]
        val_idx_src = "split"

    ns = args.tfrecord_shards
    log.info("Writing train TFRecords...")
    write_tfrecords(data, train_idx, tfdir, "train", ns, args.max_len)

    if has_val_files:
        nv_shards = max(1, ns // 4)
        write_tfrecords(vdata, np.arange(len(vdata["ids"])), tfdir, "val", nv_shards, args.max_len)
        n_val = len(vdata["ids"])
        n_test = 0
    else:
        nv_shards = max(1, ns // 4)
        write_tfrecords(data, val_idx, tfdir, "val", nv_shards, args.max_len)
        if len(test_idx) > 0:
            write_tfrecords(data, test_idx, tfdir, "test", nv_shards, args.max_len)
        n_val = len(val_idx)
        n_test = len(test_idx) if not has_val_files else 0

    # Collect all IDs for inference ordering
    np.save(os.path.join(tfdir, "ids_train.npy"), np.array([data["ids"][i] for i in train_idx], dtype=object))

    meta = {
        "n_total": n, "n_train": len(train_idx), "n_val": n_val, "n_test": n_test,
        "max_len": args.max_len, "totals": data["totals"].tolist(),
        "val_source": val_idx_src,
    }
    with open(os.path.join(tfdir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Preprocessing done. meta=%s", meta)
    return meta


def tfrecords_ready(args):
    tfdir = os.path.join(args.output_dir, "tfrecords")
    return (os.path.exists(os.path.join(tfdir, "metadata.json")) and
            len(glob.glob(os.path.join(tfdir, "train_*.tfrecord"))) > 0)


def load_meta(args):
    with open(os.path.join(args.output_dir, "tfrecords", "metadata.json")) as f:
        return json.load(f)


# =============================================================================
# TF.DATA PIPELINES
# =============================================================================
def _parse_ex(raw, max_len):
    p = tf.io.parse_single_example(raw, {
        "seq": tf.io.FixedLenFeature([], tf.string),
        "slen": tf.io.FixedLenFeature([], tf.int64),
        "freq": tf.io.FixedLenFeature([], tf.string),
        "pres": tf.io.FixedLenFeature([], tf.string),
    })
    seq = tf.cast(tf.reshape(tf.io.decode_raw(p["seq"], tf.int8), [max_len]), tf.int32)
    slen = tf.cast(p["slen"], tf.int32)
    freq = tf.reshape(tf.io.decode_raw(p["freq"], tf.float32), [3])
    pres = tf.reshape(tf.io.decode_raw(p["pres"], tf.float32), [3])
    return {"seq": seq, "slen": slen, "freq": freq, "pres": pres}


def _pad_mask(seq):
    return tf.cast(tf.not_equal(seq, PAD), tf.float32)


def _mlm_mask(parsed, max_len, mask_frac):
    """BERT-style masking: 80% MASK, 10% random, 10% keep."""
    seq = parsed["seq"]
    slen = parsed["slen"]
    pmask = _pad_mask(seq)
    rnd = tf.random.uniform([max_len])
    in_range = tf.cast(tf.range(max_len) < slen, tf.float32)
    mflag = tf.cast(rnd < mask_frac, tf.float32) * in_range
    mflag = tf.cond(tf.reduce_sum(mflag) < 1.0,
                    lambda: tf.minimum(tf.one_hot(0, max_len) * in_range + mflag, 1.0),
                    lambda: mflag)
    strat = tf.random.uniform([max_len])
    inp = tf.where(tf.logical_and(mflag > .5, strat < .8), tf.fill([max_len], MASK), seq)
    inp = tf.where(tf.logical_and(tf.logical_and(mflag > .5, strat >= .8), strat < .9),
                   tf.random.uniform([max_len], 0, N_AA, tf.int32), inp)
    return {"input_ids": inp, "original_ids": seq, "mask_flag": mflag, "padding_mask": pmask}


def _autoreg_map(parsed, max_len, mask_frac):
    """Prepare data for autoregressive VAE: MLM masking + round info."""
    seq = parsed["seq"]
    slen = parsed["slen"]
    pmask = _pad_mask(seq)
    # MLM masking for encoder
    rnd = tf.random.uniform([max_len])
    in_range = tf.cast(tf.range(max_len) < slen, tf.float32)
    mflag = tf.cast(rnd < mask_frac, tf.float32) * in_range
    mflag = tf.cond(tf.reduce_sum(mflag) < 1.0,
                    lambda: tf.minimum(tf.one_hot(0, max_len) * in_range + mflag, 1.0),
                    lambda: mflag)
    strat = tf.random.uniform([max_len])
    mlm_inp = tf.where(tf.logical_and(mflag > .5, strat < .8), tf.fill([max_len], MASK), seq)
    mlm_inp = tf.where(tf.logical_and(tf.logical_and(mflag > .5, strat >= .8), strat < .9),
                       tf.random.uniform([max_len], 0, N_AA, tf.int32), mlm_inp)
    return {
        "seq": seq, "mlm_input": mlm_inp, "mlm_mask": mflag,
        "padding_mask": pmask, "freq": parsed["freq"], "pres": parsed["pres"],
    }


def _inference_map(parsed, max_len):
    pmask = _pad_mask(parsed["seq"])
    return {"seq": parsed["seq"], "input_ids": parsed["seq"],
            "padding_mask": pmask, "freq": parsed["freq"], "pres": parsed["pres"]}


def build_dataset(args, prefix, mode, shuffle=True):
    tfdir = os.path.join(args.output_dir, "tfrecords")
    files = sorted(glob.glob(os.path.join(tfdir, f"{prefix}_*.tfrecord")))
    if not files:
        raise FileNotFoundError(f"No {prefix} TFRecords in {tfdir}")
    fds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        fds = fds.shuffle(len(files))
    ds = fds.interleave(
        lambda f: tf.data.TFRecordDataset(f, buffer_size=8*1024*1024),
        cycle_length=min(16, len(files)),
        num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)
    ml = args.max_len
    ds = ds.map(lambda x: _parse_ex(x, ml), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(min(100000, 50000))
    if mode == "mlm":
        ds = ds.map(lambda x: _mlm_mask(x, ml, args.mlm_mask_frac),
                    num_parallel_calls=tf.data.AUTOTUNE)
    elif mode == "autoreg":
        ds = ds.map(lambda x: _autoreg_map(x, ml, args.mlm_mask_frac),
                    num_parallel_calls=tf.data.AUTOTUNE)
    elif mode == "inference":
        ds = ds.map(lambda x: _inference_map(x, ml),
                    num_parallel_calls=tf.data.AUTOTUNE)
    bs = args.batch_size if mode != "inference" else args.inference_batch_size
    ds = ds.batch(bs, drop_remainder=shuffle)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# =============================================================================
# CUSTOM LAYERS
# =============================================================================
class BLOSUM62Embed(tf.keras.layers.Layer):
    """Fixed BLOSUM62 lookup → trainable projection to d_model."""
    def __init__(self, d_model, reg=None, **kw):
        super().__init__(**kw)
        self.table = tf.constant(BLOSUM62_FULL, dtype=tf.float32)
        self.proj = tf.keras.layers.Dense(d_model, use_bias=True,
                                          kernel_regularizer=reg)

    def call(self, ids):
        return self.proj(tf.gather(self.table, ids))


def _rope_freqs(max_len, hd, base=10000.0):
    half = hd // 2
    f = 1.0 / (base ** (np.arange(half, dtype=np.float32) / half))
    pos = np.arange(max_len, dtype=np.float32)
    ang = np.outer(pos, f)
    c = np.cos(np.concatenate([ang, ang], -1)).astype(np.float32)
    s = np.sin(np.concatenate([ang, ang], -1)).astype(np.float32)
    return c, s


def _rot_half(x):
    d = tf.shape(x)[-1]
    return tf.concat([-x[..., d//2:], x[..., :d//2]], -1)


def _apply_rope(x, cos, sin):
    return x * cos + _rot_half(x) * sin


class RoPEMHA(tf.keras.layers.Layer):
    """Multi-head attention with Rotary Position Embedding."""
    def __init__(self, d_model, n_heads, max_len, dropout=0.0, reg=None, **kw):
        super().__init__(**kw)
        assert d_model % n_heads == 0
        self.nh, self.hd, self.dm = n_heads, d_model // n_heads, d_model
        self.scale = self.hd ** -0.5
        self.wq = tf.keras.layers.Dense(d_model, use_bias=False, kernel_regularizer=reg)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=False, kernel_regularizer=reg)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=False, kernel_regularizer=reg)
        self.wo = tf.keras.layers.Dense(d_model, use_bias=False, kernel_regularizer=reg)
        self.drop = tf.keras.layers.Dropout(dropout)
        c, s = _rope_freqs(max_len, self.hd)
        self._cos = tf.constant(c[None, None], tf.float32)
        self._sin = tf.constant(s[None, None], tf.float32)

    def call(self, x, attn_mask=None, training=False):
        B, L = tf.shape(x)[0], tf.shape(x)[1]
        q = tf.transpose(tf.reshape(self.wq(x), [B, L, self.nh, self.hd]), [0,2,1,3])
        k = tf.transpose(tf.reshape(self.wk(x), [B, L, self.nh, self.hd]), [0,2,1,3])
        v = tf.transpose(tf.reshape(self.wv(x), [B, L, self.nh, self.hd]), [0,2,1,3])
        c = tf.cast(self._cos[:,:,:L,:], q.dtype)
        s = tf.cast(self._sin[:,:,:L,:], q.dtype)
        q, k = _apply_rope(q, c, s), _apply_rope(k, c, s)
        a = tf.matmul(q, k, transpose_b=True) * self.scale
        if attn_mask is not None:
            a += tf.cast(attn_mask, a.dtype)
        a = self.drop(tf.nn.softmax(a, -1), training=training)
        o = tf.reshape(tf.transpose(tf.matmul(a, v), [0,2,1,3]), [B, L, self.dm])
        return self.wo(o)


class LearnedPE(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kw):
        super().__init__(**kw)
        self.pe = self.add_weight("pe", [max_len, d_model], initializer="glorot_uniform")
    def call(self, x):
        return x + self.pe[None, :tf.shape(x)[1]]


class SinPE(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kw):
        super().__init__(**kw)
        p = np.zeros((max_len, d_model), np.float32)
        pos = np.arange(max_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.) / d_model))
        p[:, 0::2], p[:, 1::2] = np.sin(pos * div), np.cos(pos * div)
        self.pe = tf.constant(p[None], tf.float32)
    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1]]


class TFBlock(tf.keras.layers.Layer):
    """Pre-norm transformer block, optionally with RoPE."""
    def __init__(self, d_model, n_heads, ff_dim, dropout, use_rope=False,
                 max_len=None, reg=None, **kw):
        super().__init__(**kw)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.use_rope = use_rope
        if use_rope:
            self.mha = RoPEMHA(d_model, n_heads, max_len, dropout, reg)
        else:
            self.mha = tf.keras.layers.MultiHeadAttention(
                num_heads=n_heads, key_dim=d_model//n_heads, dropout=dropout)
        self.d1 = tf.keras.layers.Dropout(dropout)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="gelu", kernel_regularizer=reg),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model, kernel_regularizer=reg),
            tf.keras.layers.Dropout(dropout)])

    def call(self, x, padding_mask=None, training=False):
        am = None
        if padding_mask is not None:
            am = (1.0 - padding_mask[:, None, None, :]) * -1e9
        h = self.ln1(x)
        if self.use_rope:
            a = self.mha(h, attn_mask=am, training=training)
        else:
            a = self.mha(h, h, attention_mask=am, training=training)
        x = x + self.d1(a, training=training)
        return x + self.ffn(self.ln2(x), training=training)


class SeqEncoder(tf.keras.layers.Layer):
    """BLOSUM62 → [pos encoding] → N transformer blocks → mean-pool."""
    def __init__(self, d_model, n_heads, n_layers, ff_dim, dropout,
                 max_len, pos_enc="rope", reg=None, **kw):
        super().__init__(**kw)
        self._dm = d_model
        self.embed = BLOSUM62Embed(d_model, reg)
        self.use_rope = (pos_enc == "rope")
        if not self.use_rope:
            self.pe = LearnedPE(max_len, d_model) if pos_enc == "learned" else SinPE(max_len, d_model)
        self.edrop = tf.keras.layers.Dropout(dropout)
        self.blocks = [TFBlock(d_model, n_heads, ff_dim, dropout,
                               self.use_rope, max_len, reg, name=f"enc{i}")
                       for i in range(n_layers)]
        self.fln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, ids, pmask, training=False):
        x = self.embed(ids)
        if not self.use_rope:
            x = self.pe(x)
        x = self.edrop(x, training=training)
        for b in self.blocks:
            x = b(x, pmask, training=training)
        x = self.fln(x)
        m = pmask[:, :, None]
        pooled = tf.reduce_sum(x * m, 1) / (tf.reduce_sum(m, 1) + 1e-8)
        return pooled, x  # [B,d], [B,L,d]


# =============================================================================
# MLM MODEL
# =============================================================================
class MLMModel(tf.keras.Model):
    def __init__(self, args, **kw):
        super().__init__(**kw)
        reg = get_reg(args)
        self.encoder = SeqEncoder(
            args.d_model, args.n_heads, args.n_layers, args.ff_dim,
            args.dropout, args.max_len, args.pos_encoding, reg, name="encoder")
        self.mlm_head = tf.keras.Sequential([
            tf.keras.layers.Dense(args.d_model, activation="gelu", kernel_regularizer=reg),
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(N_AA, kernel_regularizer=reg),
        ], name="mlm_head")

    def call(self, inputs, training=False):
        _, per_res = self.encoder(inputs["input_ids"], inputs["padding_mask"],
                                  training=training)
        return self.mlm_head(per_res, training=training)

    def encode(self, ids, pmask, training=False):
        return self.encoder(ids, pmask, training=training)


# =============================================================================
# AUTOREGRESSIVE DECODER
# =============================================================================
class AutoregressiveDecoder(tf.keras.layers.Layer):
    """Random-order autoregressive round prediction with partial masking.

    For each sample:
      1. Pick random permutation of [R0, R1, R3].
      2. Randomly mask each round's freq+presence with autoreg_mask_rate.
      3. Decode in permutation order: for each round, predict freq & presence
         conditioned on seq_embed + all previously decoded round slots.
         If the current round is unmasked, use ground truth directly.
      4. After all 3 steps, compose canonical-order vector for VAE.

    State layout: [slot_R0(3), slot_R1(3), slot_R3(3)] = 9 dims.
      Each slot: [freq_value, presence_value, is_decoded_flag].
    """
    def __init__(self, d_model, hidden_dim, dropout=0.1, reg=None, **kw):
        super().__init__(**kw)
        # Input: seq_embed (d_model) + state (9) + target_round_onehot (3) = d_model + 12
        inp_dim = d_model + 12
        self.step_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="gelu", kernel_regularizer=reg),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(hidden_dim, activation="gelu", kernel_regularizer=reg),
            tf.keras.layers.Dense(2, kernel_regularizer=reg),  # [pred_freq, pred_pres_logit]
        ], name="autoreg_step")

    def call(self, seq_embed, freq, pres, mask_rate, training=False):
        """
        Args:
            seq_embed: [B, d_model] from MLM encoder.
            freq: [B, 3] ground truth frequencies.
            pres: [B, 3] ground truth presence (0/1).
            mask_rate: float, fraction of rounds to mask.
            training: bool.
        Returns:
            dict with autoreg_loss, decoded_freq [B,3], decoded_pres [B,3].
        """
        B = tf.shape(seq_embed)[0]

        # Random permutation per sample
        perm_idx = tf.random.uniform([B], 0, 6, tf.int32)
        perms = tf.gather(ALL_PERMS, perm_idx)  # [B, 3]

        # Random mask per round per sample: 1 = masked (predict), 0 = observed (use GT)
        round_mask = tf.cast(
            tf.random.uniform([B, 3]) < mask_rate, tf.float32)  # [B, 3]

        # State: [B, 9] = 3 slots × (freq, pres, is_decoded)
        state = tf.zeros([B, 9])
        bi = tf.range(B)

        loss_sum = tf.constant(0.0)
        n_masked = tf.constant(0.0)

        # Collect decoded values for output (in canonical R0/R1/R3 order)
        dec_freq = tf.TensorArray(tf.float32, size=3, dynamic_size=False)
        dec_pres = tf.TensorArray(tf.float32, size=3, dynamic_size=False)
        # Initialize with zeros
        for r in range(3):
            dec_freq = dec_freq.write(r, tf.zeros([B]))
            dec_pres = dec_pres.write(r, tf.zeros([B]))

        for step in range(3):
            ridx = perms[:, step]  # [B] which round
            r_oh = tf.one_hot(ridx, 3)  # [B, 3]

            # Ground truth for this round
            gt_f = tf.gather_nd(freq, tf.stack([bi, ridx], 1))  # [B]
            gt_p = tf.gather_nd(pres, tf.stack([bi, ridx], 1))  # [B]
            m = tf.gather_nd(round_mask, tf.stack([bi, ridx], 1))  # [B] 1=masked

            # Predict
            inp = tf.concat([seq_embed, state, r_oh], 1)  # [B, d+12]
            pred = self.step_net(inp, training=training)  # [B, 2]
            pf, pp_logit = pred[:, 0], pred[:, 1]

            # Loss only on masked rounds
            fl = tf.square(pf - gt_f) * m
            pl = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=gt_p, logits=pp_logit) * m
            loss_sum += tf.reduce_sum(fl + pl)
            n_masked += tf.reduce_sum(m)

            # Actual values: predicted if masked, GT if observed
            act_f = tf.where(m > 0.5, pf, gt_f)
            act_p = tf.where(m > 0.5, tf.nn.sigmoid(pp_logit), gt_p)

            # Update state slots
            new_vals = tf.stack([act_f, act_p, tf.ones([B])], 1)  # [B, 3]
            slots = [state[:, r*3:r*3+3] for r in range(3)]
            for r in range(3):
                is_r = tf.cast(tf.equal(ridx, r), tf.float32)[:, None]  # [B,1]
                slots[r] = slots[r] * (1.0 - is_r) + new_vals * is_r
            state = tf.concat(slots, 1)

            # Store decoded values in canonical order
            for r in range(3):
                is_r = tf.cast(tf.equal(ridx, r), tf.float32)  # [B]
                old_f = dec_freq.read(r)
                old_p = dec_pres.read(r)
                dec_freq = dec_freq.write(r, old_f * (1.0 - is_r) + act_f * is_r)
                dec_pres = dec_pres.write(r, old_p * (1.0 - is_r) + act_p * is_r)

        autoreg_loss = loss_sum / (n_masked + 1e-8)

        # Stack decoded values: [B, 3]
        d_f = tf.stack([dec_freq.read(r) for r in range(3)], 1)
        d_p = tf.stack([dec_pres.read(r) for r in range(3)], 1)

        return {"autoreg_loss": autoreg_loss, "dec_freq": d_f, "dec_pres": d_p}

    def predict_with_mask(self, seq_embed, freq, pres, round_mask_vec):
        """Inference: decode with specific mask vector [B, 3].
        round_mask_vec: 1=unknown/predict, 0=known/use GT.
        Uses a fixed order: known rounds first, unknown rounds last.
        """
        B = tf.shape(seq_embed)[0]
        bi = tf.range(B)

        # For simplicity during inference, use canonical order [0,1,2]
        # Known rounds get their GT values directly; unknown get predicted.
        state = tf.zeros([B, 9])
        dec_freq = [tf.zeros([B]) for _ in range(3)]
        dec_pres = [tf.zeros([B]) for _ in range(3)]

        for ridx_val in range(3):
            ridx = tf.fill([B], ridx_val)
            r_oh = tf.one_hot(ridx, 3)
            gt_f = freq[:, ridx_val]
            gt_p = pres[:, ridx_val]
            m = round_mask_vec[:, ridx_val]

            inp = tf.concat([seq_embed, state, r_oh], 1)
            pred = self.step_net(inp, training=False)
            pf, pp_logit = pred[:, 0], pred[:, 1]

            act_f = tf.where(m > 0.5, pf, gt_f)
            act_p = tf.where(m > 0.5, tf.nn.sigmoid(pp_logit), gt_p)

            new_vals = tf.stack([act_f, act_p, tf.ones([B])], 1)
            slots = [state[:, r*3:r*3+3] for r in range(3)]
            slots[ridx_val] = new_vals
            state = tf.concat(slots, 1)
            dec_freq[ridx_val] = act_f
            dec_pres[ridx_val] = act_p

        return {
            "dec_freq": tf.stack(dec_freq, 1),
            "dec_pres": tf.stack(dec_pres, 1),
        }


# =============================================================================
# VAE
# =============================================================================
class VAEEncoder(tf.keras.layers.Layer):
    """Projects [seq_embed, R0_freq, R0_pres, R1_freq, R1_pres, R3_freq, R3_pres]
       → hidden → mu, log_var → z (reparameterized)."""
    def __init__(self, d_model, hidden_dim, latent_dim, dropout=0.1, reg=None, **kw):
        super().__init__(**kw)
        inp_dim = d_model + 6  # seq_embed + 3*(freq+pres)
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="gelu", kernel_regularizer=reg),
            tf.keras.layers.Dropout(dropout),
        ], name="vae_enc_fc")
        self.mu_layer = tf.keras.layers.Dense(latent_dim, kernel_regularizer=reg, name="mu")
        self.logvar_layer = tf.keras.layers.Dense(latent_dim, kernel_regularizer=reg, name="logvar")

    def call(self, seq_embed, dec_freq, dec_pres, training=False):
        # Canonical order: [seq_embed, R0_f, R0_p, R1_f, R1_p, R3_f, R3_p]
        x = tf.concat([
            seq_embed,
            dec_freq[:, 0:1], dec_pres[:, 0:1],
            dec_freq[:, 1:2], dec_pres[:, 1:2],
            dec_freq[:, 2:3], dec_pres[:, 2:3],
        ], axis=1)
        h = self.fc(x, training=training)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        eps = tf.random.normal(tf.shape(mu))
        z = mu + tf.exp(0.5 * logvar) * eps
        return z, mu, logvar


class VAEDecoder(tf.keras.layers.Layer):
    """Decodes z → reconstructed sequence logits + round freq & presence.
    This is the 'generator' model when used standalone."""
    def __init__(self, latent_dim, hidden_dim, max_len, dropout=0.1, reg=None, **kw):
        super().__init__(**kw)
        self.max_len = max_len
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="gelu", kernel_regularizer=reg),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(hidden_dim, activation="gelu", kernel_regularizer=reg),
        ], name="vae_dec_fc")
        self.seq_head = tf.keras.layers.Dense(max_len * N_AA, kernel_regularizer=reg,
                                              name="seq_recon")
        self.round_head = tf.keras.layers.Dense(6, kernel_regularizer=reg,
                                                name="round_recon")  # 3*(freq+pres_logit)

    def call(self, z, training=False):
        h = self.fc(z, training=training)
        seq_logits = tf.reshape(self.seq_head(h), [-1, self.max_len, N_AA])
        round_out = self.round_head(h)  # [B, 6]
        # Parse: [R0_freq, R0_pres_logit, R1_freq, R1_pres_logit, R3_freq, R3_pres_logit]
        recon_freq = tf.stack([round_out[:, 0], round_out[:, 2], round_out[:, 4]], 1)  # [B,3]
        recon_pres_logit = tf.stack([round_out[:, 1], round_out[:, 3], round_out[:, 5]], 1)
        return seq_logits, recon_freq, recon_pres_logit


# =============================================================================
# FULL AUTOREG-VAE MODEL
# =============================================================================
class AutoregVAEModel(tf.keras.Model):
    """Combined: MLM encoder → Autoregressive decoder → VAE."""
    def __init__(self, args, **kw):
        super().__init__(**kw)
        reg = get_reg(args)
        self.encoder = SeqEncoder(
            args.d_model, args.n_heads, args.n_layers, args.ff_dim,
            args.dropout, args.max_len, args.pos_encoding, reg, name="encoder")
        self.mlm_head = tf.keras.Sequential([
            tf.keras.layers.Dense(args.d_model, activation="gelu", kernel_regularizer=reg),
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(N_AA, kernel_regularizer=reg),
        ], name="mlm_head")
        self.autoreg = AutoregressiveDecoder(
            args.d_model, args.autoreg_hidden_dim, args.dropout, reg, name="autoreg")
        self.vae_enc = VAEEncoder(
            args.d_model, args.vae_hidden_dim, args.vae_latent_dim,
            args.dropout, reg, name="vae_enc")
        self.vae_dec = VAEDecoder(
            args.vae_latent_dim, args.vae_hidden_dim, args.max_len,
            args.dropout, reg, name="vae_dec")
        self._args = args

    def call(self, inputs, training=False):
        """Forward for training: mlm_input → encode → autoreg → vae."""
        pooled, per_res = self.encoder(
            inputs["mlm_input"], inputs["padding_mask"], training=training)
        mlm_logits = self.mlm_head(per_res, training=training)

        ar_out = self.autoreg(
            pooled, inputs["freq"], inputs["pres"],
            self._args.autoreg_mask_rate, training=training)

        z, mu, logvar = self.vae_enc(
            pooled, ar_out["dec_freq"], ar_out["dec_pres"], training=training)
        seq_logits, recon_freq, recon_pres_logit = self.vae_dec(z, training=training)

        return {
            "mlm_logits": mlm_logits,
            "autoreg_loss": ar_out["autoreg_loss"],
            "z": z, "mu": mu, "logvar": logvar,
            "seq_logits": seq_logits,
            "recon_freq": recon_freq,
            "recon_pres_logit": recon_pres_logit,
            "dec_freq": ar_out["dec_freq"],
            "dec_pres": ar_out["dec_pres"],
        }

    def encode_to_latent(self, ids, pmask, freq, pres, round_mask_vec=None,
                         training=False):
        """Inference path: encode → autoreg (with optional masking) → VAE encode."""
        pooled, _ = self.encoder(ids, pmask, training=False)
        if round_mask_vec is None:
            round_mask_vec = tf.zeros_like(freq)  # all known
        ar = self.autoreg.predict_with_mask(pooled, freq, pres, round_mask_vec)
        z, mu, logvar = self.vae_enc(
            pooled, ar["dec_freq"], ar["dec_pres"], training=False)
        return {"z": z, "mu": mu, "pooled": pooled,
                "dec_freq": ar["dec_freq"], "dec_pres": ar["dec_pres"]}

    def generate_from_z(self, z):
        """Generator: decode from latent vector z."""
        return self.vae_dec(z, training=False)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
def mlm_loss(logits, original_ids, mask_flag):
    labels = tf.clip_by_value(tf.cast(original_ids, tf.int32), 0, 19)
    ce = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return tf.reduce_sum(ce * mask_flag) / (tf.reduce_sum(mask_flag) + 1e-8)


def vae_recon_loss(seq_logits, original_ids, padding_mask,
                   recon_freq, true_freq, recon_pres_logit, true_pres):
    """Reconstruction loss: seq CE + freq MSE + presence BCE."""
    labels = tf.clip_by_value(tf.cast(original_ids, tf.int32), 0, 19)
    seq_ce = tf.keras.losses.sparse_categorical_crossentropy(
        labels, seq_logits, from_logits=True)
    seq_loss = tf.reduce_sum(seq_ce * padding_mask) / (tf.reduce_sum(padding_mask) + 1e-8)
    freq_loss = tf.reduce_mean(tf.square(recon_freq - true_freq))
    pres_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=true_pres, logits=recon_pres_logit))
    return seq_loss, freq_loss, pres_loss


def kl_divergence(mu, logvar):
    return -0.5 * tf.reduce_mean(1.0 + logvar - tf.square(mu) - tf.exp(logvar))


# =============================================================================
# LR SCHEDULE & OPTIMIZER
# =============================================================================
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak, warmup, total):
        super().__init__()
        self.peak, self.warmup, self.total = peak, warmup, total

    def __call__(self, step):
        s = tf.cast(step, tf.float32)
        w, t = tf.cast(self.warmup, tf.float32), tf.cast(self.total, tf.float32)
        warmup_lr = s / tf.maximum(w, 1.0)
        cos_lr = 0.5 * (1.0 + tf.cos(math.pi * (s - w) / tf.maximum(t - w, 1.0)))
        return self.peak * tf.where(s < w, warmup_lr, cos_lr)

    def get_config(self):
        return {"peak": self.peak, "warmup": self.warmup, "total": self.total}


def make_optimizer(args, total_steps):
    if args.lr_schedule == "constant":
        lr = args.learning_rate
    elif args.lr_schedule == "cosine":
        lr = tf.keras.optimizers.schedules.CosineDecay(
            args.learning_rate, total_steps, alpha=1e-6)
    else:
        lr = WarmupCosine(args.learning_rate, args.warmup_steps, total_steps)
    if args.optimizer == "adamw":
        return tf.keras.optimizers.AdamW(lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return tf.keras.optimizers.SGD(lr, momentum=0.9)
    return tf.keras.optimizers.Adam(lr)


# =============================================================================
# CHECKPOINT / STATE
# =============================================================================
def _ckpt_dir(args):
    return os.path.join(args.output_dir, "checkpoints")


def save_ckpt(args, stage, model, epoch, step, best_val, history):
    d = _ckpt_dir(args)
    os.makedirs(d, exist_ok=True)
    model.save_weights(os.path.join(d, f"{stage}_latest.weights.h5"))
    state = {"epoch": epoch, "step": step, "best_val": best_val, "history": history}
    with open(os.path.join(d, f"{stage}_state.json"), "w") as f:
        json.dump(state, f, indent=2)


def save_best(args, stage, model):
    d = _ckpt_dir(args)
    model.save_weights(os.path.join(d, f"{stage}_best.weights.h5"))


def load_ckpt(args, stage):
    d = _ckpt_dir(args)
    sp = os.path.join(d, f"{stage}_state.json")
    wp = os.path.join(d, f"{stage}_latest.weights.h5")
    if os.path.exists(sp) and os.path.exists(wp):
        with open(sp) as f:
            s = json.load(f)
        s["weights_path"] = wp
        return s
    return None


def _init_model(cls, args, dummy):
    m = cls(args)
    _ = m(dummy, training=False)
    return m


def _mlm_dummy(args):
    return {"input_ids": tf.zeros([2, args.max_len], tf.int32),
            "padding_mask": tf.ones([2, args.max_len], tf.float32)}


def _autoreg_dummy(args):
    return {"seq": tf.zeros([2, args.max_len], tf.int32),
            "mlm_input": tf.zeros([2, args.max_len], tf.int32),
            "mlm_mask": tf.zeros([2, args.max_len], tf.float32),
            "padding_mask": tf.ones([2, args.max_len], tf.float32),
            "freq": tf.zeros([2, 3], tf.float32),
            "pres": tf.ones([2, 3], tf.float32)}


def _set_trainable(layer, trainable):
    """Recursively set trainable flag."""
    layer.trainable = trainable


# =============================================================================
# TRAINING: MLM
# =============================================================================
def train_mlm(args, meta):
    log.info("=" * 60)
    log.info("STAGE: MLM PRETRAINING (R0 only)")
    log.info("=" * 60)

    n_train, n_val = meta["n_train"], meta["n_val"]
    spe = max(1, n_train // args.batch_size)
    vs = max(1, n_val // args.batch_size)
    total = spe * args.epochs
    log.info("n_train=%d n_val=%d spe=%d total=%d bs=%d",
             n_train, n_val, spe, total, args.batch_size)

    train_ds = build_dataset(args, "train", "mlm", shuffle=True)
    val_ds = build_dataset(args, "val", "mlm", shuffle=False)

    model = _init_model(MLMModel, args, _mlm_dummy(args))
    model.summary(print_fn=log.info)
    opt = make_optimizer(args, total)
    gc = args.grad_clip

    # Accumulators
    loss_acc = tf.Variable(0.0, trainable=False)
    step_acc = tf.Variable(0, trainable=False, dtype=tf.int32)

    @tf.function(jit_compile=args.xla)
    def train_step(batch):
        with tf.GradientTape() as tape:
            logits = model(batch, training=True)
            loss = mlm_loss(logits, batch["original_ids"], batch["mask_flag"])
            loss += sum(model.losses)  # regularization
        gs = tape.gradient(loss, model.trainable_variables)
        gs, _ = tf.clip_by_global_norm(gs, gc)
        opt.apply_gradients(zip(gs, model.trainable_variables))
        loss_acc.assign_add(loss)
        step_acc.assign_add(1)

    @tf.function(jit_compile=args.xla)
    def val_step(batch):
        logits = model(batch, training=False)
        return mlm_loss(logits, batch["original_ids"], batch["mask_flag"])

    # Resume
    start_ep, gstep, best_val = 1, 0, float("inf")
    patience_ctr = 0
    history = {"train_loss": [], "val_loss": []}
    if args.resume:
        st = load_ckpt(args, "mlm")
        if st:
            model.load_weights(st["weights_path"])
            start_ep, gstep, best_val = st["epoch"]+1, st["step"], st["best_val"]
            history = st.get("history", history)
            log.info("Resumed MLM: ep=%d step=%d best=%.4f", start_ep-1, gstep, best_val)

    for ep in range(start_ep, args.epochs + 1):
        t0 = time.time()
        loss_acc.assign(0.0); step_acc.assign(0)
        for batch in train_ds.take(spe):
            train_step(batch)
            gstep += 1
            if gstep % args.log_every == 0:
                s = max(int(step_acc), 1)
                log.info("  step %d ep %d | loss=%.4f (%.0f s/s)",
                         gstep, ep, float(loss_acc)/s,
                         s * args.batch_size / (time.time()-t0))
            if args.ckpt_every > 0 and gstep % args.ckpt_every == 0:
                save_ckpt(args, "mlm", model, ep, gstep, best_val, history)

        tl = float(loss_acc) / max(int(step_acc), 1)
        vl, vc = 0.0, 0
        for b in val_ds.take(vs):
            vl += float(val_step(b)); vc += 1
        vl /= max(vc, 1)
        elapsed = time.time() - t0
        log.info("Epoch %d/%d train=%.4f val=%.4f (%.1fs)", ep, args.epochs, tl, vl, elapsed)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)

        if vl < best_val - args.min_delta:
            best_val, patience_ctr = vl, 0
            save_best(args, "mlm", model)
            log.info("  -> new best MLM %.4f", vl)
        else:
            patience_ctr += 1
        save_ckpt(args, "mlm", model, ep, gstep, best_val, history)
        if args.convergence and patience_ctr >= args.patience:
            log.info("Early stop at epoch %d", ep); break

    model.save_weights(os.path.join(_ckpt_dir(args), "mlm_final.weights.h5"))
    log.info("MLM done. Best val=%.4f", best_val)
    return model


# =============================================================================
# TRAINING: AUTOREGRESSIVE VAE
# =============================================================================
def train_autoreg_vae(args, meta):
    log.info("=" * 60)
    log.info("STAGE: AUTOREGRESSIVE VAE")
    log.info("=" * 60)

    n_train, n_val = meta["n_train"], meta["n_val"]
    spe = max(1, n_train // args.batch_size)
    vs = max(1, n_val // args.batch_size)
    total = spe * args.epochs
    log.info("n_train=%d n_val=%d spe=%d total=%d bs=%d",
             n_train, n_val, spe, total, args.batch_size)

    train_ds = build_dataset(args, "train", "autoreg", shuffle=True)
    val_ds = build_dataset(args, "val", "autoreg", shuffle=False)

    model = _init_model(AutoregVAEModel, args, _autoreg_dummy(args))
    model.summary(print_fn=log.info)

    # Load pretrained MLM encoder weights if available
    mlm_best = os.path.join(_ckpt_dir(args), "mlm_best.weights.h5")
    mlm_final = os.path.join(_ckpt_dir(args), "mlm_final.weights.h5")
    mlm_path = mlm_best if os.path.exists(mlm_best) else (mlm_final if os.path.exists(mlm_final) else None)
    if mlm_path:
        log.info("Loading MLM encoder from: %s", mlm_path)
        mlm_tmp = _init_model(MLMModel, args, _mlm_dummy(args))
        mlm_tmp.load_weights(mlm_path)
        for mv, ev in zip(model.encoder.trainable_variables, mlm_tmp.encoder.trainable_variables):
            mv.assign(ev)
        # Also copy MLM head
        for mv, ev in zip(model.mlm_head.trainable_variables, mlm_tmp.mlm_head.trainable_variables):
            mv.assign(ev)
        log.info("  -> transferred encoder + mlm_head weights")
        del mlm_tmp
    else:
        log.warning("No MLM weights found. Training autoreg-VAE from scratch.")

    # Apply freezing
    if args.freeze_mlm:
        _set_trainable(model.encoder, False)
        _set_trainable(model.mlm_head, False)
        log.info("Frozen: encoder + mlm_head")
    if args.freeze_autoreg:
        _set_trainable(model.autoreg, False)
        log.info("Frozen: autoreg")
    if args.freeze_vae:
        _set_trainable(model.vae_enc, False)
        _set_trainable(model.vae_dec, False)
        log.info("Frozen: vae_enc + vae_dec")

    opt = make_optimizer(args, total)
    gc = args.grad_clip
    w_rs, w_rr, w_ar, w_kl = args.w_recon_seq, args.w_recon_round, args.w_autoreg, args.kl_weight
    mf = args.mlm_mask_frac

    # Accumulators
    a_tot = tf.Variable(0.0, trainable=False)
    a_mlm = tf.Variable(0.0, trainable=False)
    a_ar = tf.Variable(0.0, trainable=False)
    a_kl = tf.Variable(0.0, trainable=False)
    a_sre = tf.Variable(0.0, trainable=False)
    a_rre = tf.Variable(0.0, trainable=False)
    a_st = tf.Variable(0, trainable=False, dtype=tf.int32)

    @tf.function(jit_compile=args.xla)
    def train_step(batch):
        with tf.GradientTape() as tape:
            out = model(batch, training=True)
            # MLM loss (auxiliary)
            lm = mlm_loss(out["mlm_logits"], batch["seq"], batch["mlm_mask"]) * mf
            # Autoreg loss
            la = out["autoreg_loss"] * w_ar
            # VAE reconstruction
            sl, fl, pl = vae_recon_loss(
                out["seq_logits"], batch["seq"], batch["padding_mask"],
                out["recon_freq"], batch["freq"],
                out["recon_pres_logit"], batch["pres"])
            l_recon_s = sl * w_rs
            l_recon_r = (fl + pl) * w_rr
            # KL
            lk = kl_divergence(out["mu"], out["logvar"]) * w_kl
            total_loss = lm + la + l_recon_s + l_recon_r + lk + sum(model.losses)
        gs = tape.gradient(total_loss, model.trainable_variables)
        gs, _ = tf.clip_by_global_norm(gs, gc)
        opt.apply_gradients(zip(gs, model.trainable_variables))
        a_tot.assign_add(total_loss); a_mlm.assign_add(lm); a_ar.assign_add(la)
        a_kl.assign_add(lk); a_sre.assign_add(l_recon_s); a_rre.assign_add(l_recon_r)
        a_st.assign_add(1)

    @tf.function(jit_compile=args.xla)
    def val_step(batch):
        out = model(batch, training=False)
        sl, fl, pl = vae_recon_loss(
            out["seq_logits"], batch["seq"], batch["padding_mask"],
            out["recon_freq"], batch["freq"],
            out["recon_pres_logit"], batch["pres"])
        lk = kl_divergence(out["mu"], out["logvar"])
        return out["autoreg_loss"]*w_ar + sl*w_rs + (fl+pl)*w_rr + lk*w_kl

    # Resume
    start_ep, gstep, best_val, patience_ctr = 1, 0, float("inf"), 0
    history = {"train_loss": [], "val_loss": [], "mlm": [], "autoreg": [],
               "kl": [], "seq_recon": [], "round_recon": []}
    if args.resume:
        st = load_ckpt(args, "autoreg_vae")
        if st:
            model.load_weights(st["weights_path"])
            start_ep, gstep, best_val = st["epoch"]+1, st["step"], st["best_val"]
            history = st.get("history", history)
            log.info("Resumed autoreg_vae: ep=%d step=%d best=%.4f",
                     start_ep-1, gstep, best_val)

    for ep in range(start_ep, args.epochs + 1):
        t0 = time.time()
        for v in [a_tot, a_mlm, a_ar, a_kl, a_sre, a_rre]:
            v.assign(0.0)
        a_st.assign(0)

        for batch in train_ds.take(spe):
            train_step(batch)
            gstep += 1
            if gstep % args.log_every == 0:
                s = max(int(a_st), 1)
                log.info("  step %d ep %d | tot=%.4f mlm=%.4f ar=%.4f kl=%.4f "
                         "srecon=%.4f rrecon=%.4f (%.0f s/s)",
                         gstep, ep, float(a_tot)/s, float(a_mlm)/s, float(a_ar)/s,
                         float(a_kl)/s, float(a_sre)/s, float(a_rre)/s,
                         s * args.batch_size / (time.time()-t0))
            if args.ckpt_every > 0 and gstep % args.ckpt_every == 0:
                save_ckpt(args, "autoreg_vae", model, ep, gstep, best_val, history)

        s = max(int(a_st), 1)
        tl = float(a_tot) / s
        vl, vc = 0.0, 0
        for b in val_ds.take(vs):
            vl += float(val_step(b)); vc += 1
        vl /= max(vc, 1)
        elapsed = time.time() - t0
        log.info("Epoch %d/%d train=%.4f val=%.4f (%.1fs)", ep, args.epochs, tl, vl, elapsed)

        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["mlm"].append(float(a_mlm)/s); history["autoreg"].append(float(a_ar)/s)
        history["kl"].append(float(a_kl)/s)
        history["seq_recon"].append(float(a_sre)/s); history["round_recon"].append(float(a_rre)/s)

        if vl < best_val - args.min_delta:
            best_val, patience_ctr = vl, 0
            save_best(args, "autoreg_vae", model)
            log.info("  -> new best autoreg_vae %.4f", vl)
        else:
            patience_ctr += 1
        save_ckpt(args, "autoreg_vae", model, ep, gstep, best_val, history)
        if args.convergence and patience_ctr >= args.patience:
            log.info("Early stop at epoch %d", ep); break

    # Save multiple model variants
    cd = _ckpt_dir(args)
    model.save_weights(os.path.join(cd, "autoreg_vae_final.weights.h5"))

    # Save encoder-only (no decoder) weights for latent extraction
    _save_encoder_only(args, model)
    # Save decoder-only weights for generator
    _save_generator(args, model)

    log.info("Autoreg-VAE done. Best val=%.4f", best_val)
    return model


def _save_encoder_only(args, model):
    """Save model weights excluding VAE decoder → 'encoder_only' checkpoint."""
    cd = _ckpt_dir(args)
    # We save the full model but mark which parts to load for encoder-only usage
    # Keras doesn't natively support partial saves cleanly, so we save component-wise
    model.encoder.save_weights(os.path.join(cd, "component_encoder.weights.h5"))
    model.mlm_head.save_weights(os.path.join(cd, "component_mlm_head.weights.h5"))
    model.autoreg.save_weights(os.path.join(cd, "component_autoreg.weights.h5"))
    model.vae_enc.save_weights(os.path.join(cd, "component_vae_enc.weights.h5"))
    model.vae_dec.save_weights(os.path.join(cd, "component_vae_dec.weights.h5"))
    log.info("Saved component weights for selective loading.")


def _save_generator(args, model):
    """Save VAE decoder as standalone generator."""
    cd = _ckpt_dir(args)
    model.vae_dec.save_weights(os.path.join(cd, "generator.weights.h5"))
    log.info("Saved generator (VAE decoder) weights.")


# =============================================================================
# INFERENCE
# =============================================================================
def run_inference(args, meta):
    log.info("=" * 60)
    log.info("INFERENCE")
    log.info("=" * 60)

    model = _init_model(AutoregVAEModel, args, _autoreg_dummy(args))

    # Try loading best, then final
    cd = _ckpt_dir(args)
    for wn in ["autoreg_vae_best.weights.h5", "autoreg_vae_final.weights.h5",
               "autoreg_vae_latest.weights.h5"]:
        wp = os.path.join(cd, wn)
        if os.path.exists(wp):
            model.load_weights(wp)
            log.info("Loaded weights: %s", wp)
            break
    else:
        log.error("No autoreg_vae weights found in %s", cd)
        return

    # Parse mask rounds
    mask_rounds = set()
    if args.inference_mask_rounds:
        for r in args.inference_mask_rounds.split(","):
            r = r.strip().upper()
            if r in ROUND_NAMES:
                mask_rounds.add(ROUND_NAMES.index(r))
    log.info("Masking rounds: %s", [ROUND_NAMES[i] for i in mask_rounds])

    out_dir = os.path.join(args.output_dir, "inference")
    os.makedirs(out_dir, exist_ok=True)

    results = {k: [] for k in ["z", "mu", "dec_freq", "dec_pres", "pooled"]}
    total = 0

    for prefix in ["train", "val", "test"]:
        tfdir = os.path.join(args.output_dir, "tfrecords")
        if not glob.glob(os.path.join(tfdir, f"{prefix}_*.tfrecord")):
            continue
        ds = build_dataset(args, prefix, "inference", shuffle=False)
        for batch in ds:
            B = tf.shape(batch["input_ids"])[0]
            rmask = tf.zeros([B, 3])
            if mask_rounds:
                cols = []
                for r in range(3):
                    cols.append(tf.ones([B]) if r in mask_rounds else tf.zeros([B]))
                rmask = tf.stack(cols, 1)

            out = model.encode_to_latent(
                batch["input_ids"], batch["padding_mask"],
                batch["freq"], batch["pres"], rmask)
            for k in results:
                results[k].append(out[k].numpy())
            total += int(B)
            if total % (args.inference_batch_size * 100) < args.inference_batch_size:
                log.info("  processed %d", total)

    for k in results:
        results[k] = np.concatenate(results[k], 0)
    N = results["z"].shape[0]
    log.info("Total: %d sequences", N)

    # Save
    np.savez_compressed(os.path.join(out_dir, "latents.npz"), **results)

    # TSV with predictions
    ids_path = os.path.join(args.output_dir, "tfrecords", "ids_train.npy")
    ids = np.load(ids_path, allow_pickle=True) if os.path.exists(ids_path) else None
    with open(os.path.join(out_dir, "predictions.tsv"), "w") as f:
        f.write("seq_id\tz_norm\tdec_freq_R0\tdec_freq_R1\tdec_freq_R3\t"
                "dec_pres_R0\tdec_pres_R1\tdec_pres_R3\n")
        for i in range(N):
            sid = ids[i] if ids is not None and i < len(ids) else str(i)
            zn = np.linalg.norm(results["z"][i])
            f.write(f"{sid}\t{zn:.6f}\t"
                    f"{results['dec_freq'][i,0]:.6f}\t{results['dec_freq'][i,1]:.6f}\t"
                    f"{results['dec_freq'][i,2]:.6f}\t"
                    f"{results['dec_pres'][i,0]:.4f}\t{results['dec_pres'][i,1]:.4f}\t"
                    f"{results['dec_pres'][i,2]:.4f}\n")
    log.info("Saved inference results to %s", out_dir)


# =============================================================================
# GENERATOR
# =============================================================================
def run_generator(args, meta):
    log.info("=" * 60)
    log.info("GENERATOR: sample from VAE latent space")
    log.info("=" * 60)

    model = _init_model(AutoregVAEModel, args, _autoreg_dummy(args))
    cd = _ckpt_dir(args)
    for wn in ["autoreg_vae_best.weights.h5", "autoreg_vae_final.weights.h5"]:
        wp = os.path.join(cd, wn)
        if os.path.exists(wp):
            model.load_weights(wp)
            log.info("Loaded: %s", wp)
            break
    else:
        log.error("No weights found"); return

    out_dir = os.path.join(args.output_dir, "generated")
    os.makedirs(out_dir, exist_ok=True)

    n = args.n_generate
    z = tf.random.normal([n, args.vae_latent_dim])
    seq_logits, recon_freq, recon_pres_logit = model.generate_from_z(z)

    seq_ids = tf.argmax(seq_logits, -1).numpy()  # [N, L]
    recon_freq_np = recon_freq.numpy()
    recon_pres_np = tf.nn.sigmoid(recon_pres_logit).numpy()

    idx2aa = {i: a for a, i in AA2I.items()}
    with open(os.path.join(out_dir, "generated.fasta"), "w") as ff:
        with open(os.path.join(out_dir, "generated_rounds.tsv"), "w") as ft:
            ft.write("gen_id\tfreq_R0\tfreq_R1\tfreq_R3\tpres_R0\tpres_R1\tpres_R3\n")
            for i in range(n):
                seq = "".join(idx2aa.get(int(aa), "X") for aa in seq_ids[i] if int(aa) < N_AA)
                ff.write(f">gen_{i}\n{seq}\n")
                ft.write(f"gen_{i}\t{recon_freq_np[i,0]:.6f}\t{recon_freq_np[i,1]:.6f}\t"
                         f"{recon_freq_np[i,2]:.6f}\t{recon_pres_np[i,0]:.4f}\t"
                         f"{recon_pres_np[i,1]:.4f}\t{recon_pres_np[i,2]:.4f}\n")
    log.info("Generated %d sequences -> %s", n, out_dir)


# =============================================================================
# ANALYSIS
# =============================================================================
def run_analysis(args, meta):
    log.info("=" * 60)
    log.info("ANALYSIS")
    log.info("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
    except ImportError:
        log.error("matplotlib and scikit-learn required for --analysis")
        return

    out_dir = os.path.join(args.output_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Training curves ----
    for stage in ["mlm", "autoreg_vae"]:
        sp = os.path.join(_ckpt_dir(args), f"{stage}_state.json")
        if not os.path.exists(sp):
            continue
        with open(sp) as f:
            st = json.load(f)
        hist = st.get("history", {})
        if "train_loss" in hist and hist["train_loss"]:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(hist["train_loss"], label="train")
            if "val_loss" in hist:
                ax.plot(hist["val_loss"], label="val")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.set_title(f"{stage} Training Curves"); ax.legend()
            fig.savefig(os.path.join(out_dir, f"{stage}_curves.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("Saved %s curves", stage)

        # Component losses for autoreg_vae
        if stage == "autoreg_vae":
            comps = ["mlm", "autoreg", "kl", "seq_recon", "round_recon"]
            avail = [c for c in comps if c in hist and hist[c]]
            if avail:
                fig, ax = plt.subplots(figsize=(10, 5))
                for c in avail:
                    ax.plot(hist[c], label=c)
                ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
                ax.set_title("Autoreg-VAE Component Losses"); ax.legend()
                fig.savefig(os.path.join(out_dir, "autoreg_vae_components.png"),
                            dpi=150, bbox_inches="tight")
                plt.close(fig)

    # ---- AUC / PR per round (requires inference results) ----
    lat_path = os.path.join(args.output_dir, "inference", "latents.npz")
    if not os.path.exists(lat_path):
        log.info("Running inference first for AUC/PR analysis...")
        run_inference(args, meta)

    if os.path.exists(lat_path):
        lat = np.load(lat_path)
        # We need ground truth presence. Reload from TFRecords.
        gt_pres_list = []
        pred_pres_list = []

        for prefix in ["train", "val", "test"]:
            tfdir = os.path.join(args.output_dir, "tfrecords")
            if not glob.glob(os.path.join(tfdir, f"{prefix}_*.tfrecord")):
                continue
            ds = build_dataset(args, prefix, "inference", shuffle=False)
            for batch in ds:
                gt_pres_list.append(batch["pres"].numpy())

        if gt_pres_list:
            gt_pres = np.concatenate(gt_pres_list, 0)
            pred_pres = lat["dec_pres"]
            n_samples = min(len(gt_pres), len(pred_pres))
            gt_pres = gt_pres[:n_samples]
            pred_pres = pred_pres[:n_samples]

            for ri, rname in enumerate(ROUND_NAMES):
                y_true = gt_pres[:, ri]
                y_score = pred_pres[:, ri]
                if len(np.unique(y_true)) < 2:
                    log.info("  %s: only one class, skipping AUC/PR", rname)
                    continue

                auc = roc_auc_score(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
                log.info("  %s: AUC=%.4f  AP=%.4f", rname, auc, ap)

                # PR curve
                prec, rec, _ = precision_recall_curve(y_true, y_score)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(rec, prec)
                ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
                ax.set_title(f"{rname} PR Curve (AP={ap:.3f})")
                fig.savefig(os.path.join(out_dir, f"pr_{rname}.png"),
                            dpi=150, bbox_inches="tight")
                plt.close(fig)

            # ---- UMAP of VAE latent space ----
            try:
                import umap
                log.info("Computing UMAP projection of VAE latent space...")
                z = lat["z"] if "z" in lat else lat.get("mu", None)
                if z is not None and len(z) > 100:
                    # Subsample for speed
                    max_pts = min(50000, len(z))
                    idx = np.random.choice(len(z), max_pts, replace=False)
                    emb = umap.UMAP(n_components=2, random_state=args.seed).fit_transform(z[idx])
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    for ri, rname in enumerate(ROUND_NAMES):
                        sc = axes[ri].scatter(emb[:, 0], emb[:, 1],
                                              c=gt_pres[idx, ri], cmap="RdYlGn",
                                              s=1, alpha=0.5)
                        axes[ri].set_title(f"UMAP colored by {rname} presence")
                        plt.colorbar(sc, ax=axes[ri])
                    fig.tight_layout()
                    fig.savefig(os.path.join(out_dir, "umap_latent.png"),
                                dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    np.savez_compressed(os.path.join(out_dir, "umap_coords.npz"),
                                        coords=emb, indices=idx)
                    log.info("Saved UMAP plot")
            except ImportError:
                log.warning("umap-learn not installed, skipping UMAP.")

    log.info("Analysis complete -> %s", out_dir)


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    set_seeds(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        log.info("Mixed precision enabled")
    if args.xla:
        log.info("XLA enabled")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        log.info("GPUs: %s", [g.name for g in gpus])
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    else:
        log.info("No GPU — CPU mode")

    # Save args
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # TFRecord preprocessing
    if tfrecords_ready(args) and not args.force_tfrecord:
        log.info("TFRecords found — skipping preprocessing")
        meta = load_meta(args)
    else:
        log.info("Preprocessing FASTA → TFRecords...")
        meta = preprocess_data(args)

    log.info("Data: n_total=%d n_train=%d n_val=%d n_test=%d",
             meta["n_total"], meta["n_train"], meta["n_val"], meta["n_test"])

    # Stage dispatch
    if args.mlm:
        train_mlm(args, meta)

    if args.autoregressive_vae:
        train_autoreg_vae(args, meta)

    if args.inference:
        run_inference(args, meta)

    if args.generator:
        run_generator(args, meta)

    if args.analysis:
        run_analysis(args, meta)

    if not any([args.mlm, args.autoregressive_vae, args.inference,
                args.generator, args.analysis]):
        log.info("No stage selected. Use --mlm, --autoregressive_vae, "
                 "--inference, --generator, or --analysis.")


if __name__ == "__main__":
    main()