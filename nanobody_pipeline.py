#!/usr/bin/env python3
"""
================================================================================
Nanobody Phage Display Pipeline — MLM + Autoregressive VAE (v3.2)
================================================================================
End-to-end pipeline:
  Stage 1 — MLM: Masked language model pretraining on R0 sequences.
  Stage 2 — Autoregressive VAE: Random-order autoregressive decoding of
            round presence/frequency/enrichment with partial masking,
            followed by VAE reconstruction.

Enrichment:
  E_R0 = 1.0 (identity), E_R1 = freq_R1/freq_R0, E_R3 = freq_R3/freq_R0.

Usage examples:
  python pipeline.py --r0_train r0.fa --r1_train r1.fa --r3_train r3.fa \
      --output_dir out --mlm --epochs 20

  python pipeline.py --r0_train r0.fa --r1_train r1.fa --r3_train r3.fa \
      --output_dir out --autoregressive_vae --epochs 50

  python pipeline.py --r0_train r0.fa --r1_train r1.fa --r3_train r3.fa \
      --output_dir out --inference --inference_mask_rounds R1,R3

  python pipeline.py --r0_train r0.fa --r1_train r1.fa --r3_train r3.fa \
      --output_dir out --generator --n_generate 1000

  python pipeline.py --r0_train r0.fa --r1_train r1.fa --r3_train r3.fa \
      --output_dir out --analysis
================================================================================
"""

import os, sys, glob, math, time, json, random, logging, argparse
import numpy as np

# --- GPU config MUST happen before any TF op triggers context init ---
import tensorflow as tf
_gpus = tf.config.list_physical_devices("GPU")
for _g in _gpus:
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except RuntimeError:
        pass  # already initialized

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
ENRICH_EPS = 1e-8   # epsilon for enrichment division
ENRICH_CLIP = 100.0 # max enrichment ratio (prevents NaN from tiny freq_R0)
LOGVAR_CLIP = 10.0  # max logvar in VAE (exp(5)=148, safe from overflow)

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
BLOSUM62_FULL = np.vstack([BLOSUM62_NORM, np.zeros((3, 20), dtype=np.float32)])

ALL_PERMS = tf.constant([
    [0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]
], dtype=tf.int32)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Nanobody Pipeline v3: MLM + Autoregressive VAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("Data")
    g.add_argument("--r0_train", required=True); g.add_argument("--r1_train", required=True)
    g.add_argument("--r3_train", required=True)
    g.add_argument("--r0_val", default=None); g.add_argument("--r1_val", default=None)
    g.add_argument("--r3_val", default=None)
    g.add_argument("--val_split", type=float, default=0.1)
    g.add_argument("--test_split", type=float, default=0.05)
    g.add_argument("--max_len", type=int, default=141)
    g.add_argument("--pseudocount", type=float, default=0.5)
    g.add_argument("--tfrecord_shards", type=int, default=32)
    g.add_argument("--force_tfrecord", action="store_true")
    p.add_argument("--output_dir", required=True)
    g2 = p.add_argument_group("Pipeline stages")
    g2.add_argument("--mlm", action="store_true")
    g2.add_argument("--autoregressive_vae", action="store_true")
    g2.add_argument("--inference", action="store_true")
    g2.add_argument("--generator", action="store_true")
    g2.add_argument("--analysis", action="store_true")
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
    g4 = p.add_argument_group("Autoregressive VAE")
    g4.add_argument("--log_enrichment", action="store_true",
                    help="Use log(enrichment+eps) instead of raw enrichment ratio.")
    g4.add_argument("--autoreg_hidden_dim", type=int, default=64)
    g4.add_argument("--autoreg_mask_rate", type=float, default=0.5)
    g4.add_argument("--vae_hidden_dim", type=int, default=64)
    g4.add_argument("--vae_latent_dim", type=int, default=64)
    g4.add_argument("--kl_weight", type=float, default=1e-3)
    g4.add_argument("--w_recon_seq", type=float, default=1.0)
    g4.add_argument("--w_recon_round", type=float, default=1.0)
    g4.add_argument("--w_autoreg", type=float, default=1.0)
    g5 = p.add_argument_group("Training")
    g5.add_argument("--epochs", type=int, default=50)
    g5.add_argument("--batch_size", type=int, default=256)
    g5.add_argument("--learning_rate", type=float, default=1e-3)
    g5.add_argument("--optimizer", default="adam", choices=["adam","adamw","sgd"])
    g5.add_argument("--weight_decay", type=float, default=0.0)
    g5.add_argument("--lr_schedule", default="warmup_cosine", choices=["constant","cosine","warmup_cosine"])
    g5.add_argument("--warmup_steps", type=int, default=500)
    g5.add_argument("--grad_clip", type=float, default=1.0)
    g5.add_argument("--convergence", action="store_true")
    g5.add_argument("--patience", type=int, default=5)
    g5.add_argument("--min_delta", type=float, default=1e-4)
    g5.add_argument("--log_every", type=int, default=100)
    g5.add_argument("--ckpt_every", type=int, default=1000)
    g5.add_argument("--resume", action="store_true")
    g6 = p.add_argument_group("Freezing")
    g6.add_argument("--freeze_mlm", action="store_true")
    g6.add_argument("--freeze_autoreg", action="store_true")
    g6.add_argument("--freeze_vae", action="store_true")
    g7 = p.add_argument_group("Inference / Generator")
    g7.add_argument("--inference_batch_size", type=int, default=512)
    g7.add_argument("--inference_mask_rounds", type=str, default="",
                    help="Comma-sep rounds to mask, e.g. 'R1,R3'.")
    g7.add_argument("--n_generate", type=int, default=100)
    g8 = p.add_argument_group("Performance")
    g8.add_argument("--mixed_precision", action="store_true")
    g8.add_argument("--xla", action="store_true")
    g8.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seeds(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)


def get_reg(args):
    if args.l1_reg > 0 and args.l2_reg > 0:
        return tf.keras.regularizers.L1L2(args.l1_reg, args.l2_reg)
    if args.l1_reg > 0: return tf.keras.regularizers.L1(args.l1_reg)
    if args.l2_reg > 0: return tf.keras.regularizers.L2(args.l2_reg)
    return None


def compute_enrichment(freq, use_log=False):
    """Compute enrichment from freq [B,3]. E_R0=1, E_R1=f1/f0, E_R3=f3/f0.
    Clipped to [0, ENRICH_CLIP] to prevent NaN from tiny freq_R0.
    If use_log, returns log(E + eps) so E_R0≈0, E_R1=log(f1/f0+eps), etc."""
    f0 = tf.maximum(freq[:, 0:1], ENRICH_EPS)
    e0 = tf.ones_like(freq[:, 0:1])
    e1 = tf.clip_by_value(freq[:, 1:2] / f0, 0.0, ENRICH_CLIP)
    e3 = tf.clip_by_value(freq[:, 2:3] / f0, 0.0, ENRICH_CLIP)
    e = tf.concat([e0, e1, e3], axis=1)  # [B, 3]
    if use_log:
        e = tf.math.log(e + ENRICH_EPS)  # E_R0 -> log(1+eps) ≈ 0
    return e


# =============================================================================
# DATA LOADING & TFRECORD I/O
# =============================================================================
def parse_fasta(path):
    sid, cnt, parts = None, None, []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            if ln.startswith(">"):
                if sid is not None: yield sid, cnt, "".join(parts)
                toks = ln[1:].rsplit("_", 1)
                sid, cnt, parts = toks[0], float(toks[1]), []
            else: parts.append(ln.upper())
    if sid is not None: yield sid, cnt, "".join(parts)


def load_round_files(r0p, r1p, r3p, max_len, pseudocount=0.5):
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
        if (i + 1) % 2_000_000 == 0: log.info("  encoded %d/%d", i + 1, n)
    freq = (counts / tot[None, :].astype(np.float32)).astype(np.float32)
    pres = (counts > 0).astype(np.float32)
    pres[:, 0] = np.where(counts[:, 0] > pseudocount, 1.0, 0.0)
    log.info("Encoded %d sequences", n)
    return {"seqs": seqs, "slens": slens, "counts": counts, "freq": freq,
            "pres": pres, "ids": all_ids, "totals": tot}


def _bf(v): return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
def _if(v): return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def write_tfrecords(data, indices, out_dir, prefix, n_shards, max_len):
    os.makedirs(out_dir, exist_ok=True)
    n = len(indices); per = (n + n_shards - 1) // n_shards; written = 0
    for sh in range(n_shards):
        s, e = sh * per, min((sh + 1) * per, n)
        if s >= n: break
        path = os.path.join(out_dir, f"{prefix}_{sh:05d}.tfrecord")
        with tf.io.TFRecordWriter(path) as w:
            for idx in indices[s:e]:
                feat = {"seq": _bf(data["seqs"][idx].tobytes()),
                        "slen": _if(int(data["slens"][idx])),
                        "freq": _bf(data["freq"][idx].tobytes()),
                        "pres": _bf(data["pres"][idx].tobytes())}
                w.write(tf.train.Example(
                    features=tf.train.Features(feature=feat)).SerializeToString())
                written += 1
    log.info("  %s: %d records in %d shards", prefix, written, n_shards)
    return written


def preprocess_data(args):
    tfdir = os.path.join(args.output_dir, "tfrecords")
    data = load_round_files(args.r0_train, args.r1_train, args.r3_train,
                            args.max_len, args.pseudocount)
    n = len(data["ids"])
    has_val = (args.r0_val and args.r1_val and args.r3_val)
    if has_val:
        vdata = load_round_files(args.r0_val, args.r1_val, args.r3_val,
                                 args.max_len, args.pseudocount)
        train_idx = np.arange(n); val_src = "external"
    else:
        rng = np.random.RandomState(args.seed); perm = rng.permutation(n)
        nt = int(n * args.test_split); nv = int(n * args.val_split)
        test_idx, val_idx, train_idx = perm[:nt], perm[nt:nt+nv], perm[nt+nv:]
        val_src = "split"
    ns = args.tfrecord_shards
    log.info("Writing train TFRecords...")
    write_tfrecords(data, train_idx, tfdir, "train", ns, args.max_len)
    if has_val:
        write_tfrecords(vdata, np.arange(len(vdata["ids"])), tfdir, "val",
                        max(1, ns//4), args.max_len)
        n_val, n_test = len(vdata["ids"]), 0
    else:
        write_tfrecords(data, val_idx, tfdir, "val", max(1, ns//4), args.max_len)
        if len(test_idx) > 0:
            write_tfrecords(data, test_idx, tfdir, "test", max(1, ns//4), args.max_len)
        n_val, n_test = len(val_idx), len(test_idx)
    np.save(os.path.join(tfdir, "ids_train.npy"),
            np.array([data["ids"][i] for i in train_idx], dtype=object))
    meta = {"n_total": n, "n_train": len(train_idx), "n_val": n_val,
            "n_test": n_test, "max_len": args.max_len,
            "totals": data["totals"].tolist(), "val_source": val_src}
    with open(os.path.join(tfdir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Preprocessing done.")
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
    seq, slen = parsed["seq"], parsed["slen"]
    pmask = _pad_mask(seq)
    rnd = tf.random.uniform([max_len])
    in_range = tf.cast(tf.range(max_len) < slen, tf.float32)
    mflag = tf.cast(rnd < mask_frac, tf.float32) * in_range
    mflag = tf.cond(tf.reduce_sum(mflag) < 1.0,
                    lambda: tf.minimum(tf.one_hot(0, max_len)*in_range + mflag, 1.0),
                    lambda: mflag)
    strat = tf.random.uniform([max_len])
    inp = tf.where(tf.logical_and(mflag > .5, strat < .8), tf.fill([max_len], MASK), seq)
    inp = tf.where(tf.logical_and(tf.logical_and(mflag > .5, strat >= .8), strat < .9),
                   tf.random.uniform([max_len], 0, N_AA, tf.int32), inp)
    return {"input_ids": inp, "original_ids": seq, "mask_flag": mflag, "padding_mask": pmask}


def _autoreg_map(parsed, max_len, mask_frac):
    seq, slen = parsed["seq"], parsed["slen"]
    pmask = _pad_mask(seq)
    rnd = tf.random.uniform([max_len])
    in_range = tf.cast(tf.range(max_len) < slen, tf.float32)
    mflag = tf.cast(rnd < mask_frac, tf.float32) * in_range
    mflag = tf.cond(tf.reduce_sum(mflag) < 1.0,
                    lambda: tf.minimum(tf.one_hot(0, max_len)*in_range + mflag, 1.0),
                    lambda: mflag)
    strat = tf.random.uniform([max_len])
    mlm_inp = tf.where(tf.logical_and(mflag > .5, strat < .8), tf.fill([max_len], MASK), seq)
    mlm_inp = tf.where(tf.logical_and(tf.logical_and(mflag > .5, strat >= .8), strat < .9),
                       tf.random.uniform([max_len], 0, N_AA, tf.int32), mlm_inp)
    return {"seq": seq, "mlm_input": mlm_inp, "mlm_mask": mflag,
            "padding_mask": pmask, "freq": parsed["freq"], "pres": parsed["pres"]}


def _inference_map(parsed, max_len):
    pmask = _pad_mask(parsed["seq"])
    return {"seq": parsed["seq"], "input_ids": parsed["seq"],
            "padding_mask": pmask, "freq": parsed["freq"], "pres": parsed["pres"]}


def build_dataset(args, prefix, mode, shuffle=True):
    tfdir = os.path.join(args.output_dir, "tfrecords")
    files = sorted(glob.glob(os.path.join(tfdir, f"{prefix}_*.tfrecord")))
    if not files: raise FileNotFoundError(f"No {prefix} TFRecords in {tfdir}")
    fds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle: fds = fds.shuffle(len(files))
    ds = fds.interleave(
        lambda f: tf.data.TFRecordDataset(f, buffer_size=8*1024*1024),
        cycle_length=min(16, len(files)),
        num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)
    ml = args.max_len
    ds = ds.map(lambda x: _parse_ex(x, ml), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle: ds = ds.shuffle(min(100000, 50000))
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
    ds = ds.batch(bs, drop_remainder=shuffle).prefetch(tf.data.AUTOTUNE)
    return ds


# =============================================================================
# CUSTOM LAYERS
# =============================================================================
class BLOSUM62Embed(tf.keras.layers.Layer):
    def __init__(self, d_model, reg=None, **kw):
        super().__init__(**kw)
        self.table = tf.constant(BLOSUM62_FULL, dtype=tf.float32)
        self.proj = tf.keras.layers.Dense(d_model, use_bias=True, kernel_regularizer=reg)
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
    def __init__(self, d_model, n_heads, max_len, dropout=0.0, reg=None, **kw):
        super().__init__(**kw)
        assert d_model % n_heads == 0
        self.nh, self.hd, self.dm = n_heads, d_model//n_heads, d_model
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
        if attn_mask is not None: a += tf.cast(attn_mask, a.dtype)
        a = self.drop(tf.nn.softmax(a, -1), training=training)
        o = tf.reshape(tf.transpose(tf.matmul(a, v), [0,2,1,3]), [B, L, self.dm])
        return self.wo(o)


class LearnedPE(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kw):
        super().__init__(**kw)
        self.pe = self.add_weight("pe", [max_len, d_model], initializer="glorot_uniform")
    def call(self, x): return x + self.pe[None, :tf.shape(x)[1]]


class SinPE(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kw):
        super().__init__(**kw)
        p = np.zeros((max_len, d_model), np.float32)
        pos = np.arange(max_len)[:,None]
        div = np.exp(np.arange(0, d_model, 2)*-(np.log(10000.)/d_model))
        p[:,0::2], p[:,1::2] = np.sin(pos*div), np.cos(pos*div)
        self.pe = tf.constant(p[None], tf.float32)
    def call(self, x): return x + self.pe[:,:tf.shape(x)[1]]


class TFBlock(tf.keras.layers.Layer):
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
    def __init__(self, d_model, n_heads, n_layers, ff_dim, dropout,
                 max_len, pos_enc="rope", reg=None, **kw):
        super().__init__(**kw)
        self._dm = d_model
        self.embed = BLOSUM62Embed(d_model, reg)
        self.use_rope = (pos_enc == "rope")
        if not self.use_rope:
            self.pe = LearnedPE(max_len, d_model) if pos_enc=="learned" else SinPE(max_len, d_model)
        self.edrop = tf.keras.layers.Dropout(dropout)
        self.blocks = [TFBlock(d_model, n_heads, ff_dim, dropout,
                               self.use_rope, max_len, reg, name=f"enc{i}")
                       for i in range(n_layers)]
        self.fln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, ids, pmask, training=False):
        x = self.embed(ids)
        if not self.use_rope: x = self.pe(x)
        x = self.edrop(x, training=training)
        for b in self.blocks: x = b(x, pmask, training=training)
        x = self.fln(x)
        m = pmask[:,:,None]
        pooled = tf.reduce_sum(x*m, 1) / (tf.reduce_sum(m, 1) + 1e-8)
        return pooled, x


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
        _, per_res = self.encoder(inputs["input_ids"], inputs["padding_mask"], training=training)
        return self.mlm_head(per_res, training=training)

    def encode(self, ids, pmask, training=False):
        return self.encoder(ids, pmask, training=training)


# =============================================================================
# AUTOREGRESSIVE DECODER (with enrichment)
# =============================================================================
class AutoregressiveDecoder(tf.keras.layers.Layer):
    """Random-order autoregressive decoder with separated heads.

    Two independent prediction paths to prevent information leakage:

    1. Presence head (pres_net):
       Input: seq_embed + pres_state (6) + round_onehot (3)
       Only sees: sequence embedding, round identity, and previously decoded
       presence probabilities. NO access to frequency or enrichment.
       Output: presence logit (1 value).

    2. Frequency/Enrichment head (fe_net):
       Input: seq_embed + full_state (12) + round_onehot (3)
       Sees everything: sequence embedding, round identity, and all previously
       decoded values (presence, frequency, enrichment).
       Output: frequency and enrichment (2 values).

    This separation ensures presence is predicted from sequence content and
    round context only, forcing the model to learn binding-relevant features
    rather than shortcuts through frequency statistics.

    State layouts:
      pres_state: [R0_pres, R0_decoded, R1_pres, R1_decoded, R3_pres, R3_decoded] = 6
      full_state: [R0_freq, R0_pres, R0_enrich, R0_decoded, R1_..., R3_...] = 12
    """
    PRES_SLOT = 2   # per-round in pres_state: [pres_val, is_decoded]
    FULL_SLOT = 4   # per-round in full_state: [freq, pres, enrich, is_decoded]
    N_ROUNDS = 3
    PRES_STATE_DIM = PRES_SLOT * N_ROUNDS   # 6
    FULL_STATE_DIM = FULL_SLOT * N_ROUNDS   # 12

    def __init__(self, d_model, hidden_dim, mask_rate=0.5, dropout=0.1,
                 reg=None, **kw):
        super().__init__(**kw)
        self.mask_rate = mask_rate

        # Presence head: seq_embed + pres_state(6) + round_oh(3) -> 1
        self.pres_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="gelu", kernel_regularizer=reg),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(hidden_dim // 2, activation="gelu", kernel_regularizer=reg),
            tf.keras.layers.Dense(1, kernel_regularizer=reg),
        ], name="pres_net")

        # Freq/enrichment head: seq_embed + full_state(12) + round_oh(3) -> 2
        self.fe_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="gelu", kernel_regularizer=reg),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(hidden_dim, activation="gelu", kernel_regularizer=reg),
            tf.keras.layers.Dense(2, kernel_regularizer=reg),  # [freq, enrich]
        ], name="fe_net")

    def _decode_steps(self, seq_embed, freq, pres, enrich, round_mask, perms,
                      training):
        """Core autoregressive loop with separated heads.

        For each step in the permutation order:
          1. pres_net predicts presence from (seq_embed, pres_state, round_oh)
          2. fe_net predicts freq+enrich from (seq_embed, full_state, round_oh)
          3. Both states are updated with decoded or ground-truth values.

        Returns:
            loss_freq, loss_pres, loss_enrich: scalar losses (masked only).
            dec_freq, dec_pres, dec_enrich: [B,3] decoded values.
        """
        B = tf.shape(seq_embed)[0]
        bi = tf.range(B)
        PS, FS = self.PRES_SLOT, self.FULL_SLOT

        pres_state = tf.zeros([B, self.PRES_STATE_DIM])   # [B, 6]
        full_state = tf.zeros([B, self.FULL_STATE_DIM])    # [B, 12]

        loss_freq = tf.constant(0.0)
        loss_pres = tf.constant(0.0)
        loss_enrich = tf.constant(0.0)
        n_masked = tf.constant(0.0)

        dec_freq = tf.TensorArray(tf.float32, size=3)
        dec_pres = tf.TensorArray(tf.float32, size=3)
        dec_enrich = tf.TensorArray(tf.float32, size=3)
        for r in range(3):
            dec_freq = dec_freq.write(r, tf.zeros([B]))
            dec_pres = dec_pres.write(r, tf.zeros([B]))
            dec_enrich = dec_enrich.write(r, tf.zeros([B]))

        for step in range(3):
            ridx = perms[:, step]
            r_oh = tf.one_hot(ridx, 3)  # [B, 3]

            gt_f = tf.gather_nd(freq, tf.stack([bi, ridx], 1))
            gt_p = tf.gather_nd(pres, tf.stack([bi, ridx], 1))
            gt_e = tf.gather_nd(enrich, tf.stack([bi, ridx], 1))
            m = tf.gather_nd(round_mask, tf.stack([bi, ridx], 1))

            # --- Presence prediction (isolated: no freq/enrich info) ---
            pres_inp = tf.concat([seq_embed, pres_state, r_oh], 1)  # [B, d+9]
            pp_logit = self.pres_net(pres_inp, training=training)[:, 0]  # [B]

            # --- Freq/Enrichment prediction (full context) ---
            fe_inp = tf.concat([seq_embed, full_state, r_oh], 1)  # [B, d+15]
            fe_pred = self.fe_net(fe_inp, training=training)  # [B, 2]
            pf, pe = fe_pred[:, 0], fe_pred[:, 1]

            # Losses (masked positions only)
            loss_freq += tf.reduce_sum(tf.square(pf - gt_f) * m)
            loss_pres += tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_p, logits=pp_logit) * m)
            loss_enrich += tf.reduce_sum(tf.square(pe - gt_e) * m)
            n_masked += tf.reduce_sum(m)

            # Actual values: predicted if masked, GT if observed (clipped)
            act_f = tf.where(m > 0.5, tf.clip_by_value(pf, -10.0, 10.0), gt_f)
            act_p = tf.where(m > 0.5, tf.nn.sigmoid(pp_logit), gt_p)
            act_e = tf.where(m > 0.5, tf.clip_by_value(pe, -ENRICH_CLIP, ENRICH_CLIP), gt_e)

            # Update presence-only state: [pres_val, is_decoded] per round
            new_pres_slot = tf.stack([act_p, tf.ones([B])], 1)  # [B, 2]
            p_slots = [pres_state[:, r*PS:(r+1)*PS] for r in range(3)]
            for r in range(3):
                is_r = tf.cast(tf.equal(ridx, r), tf.float32)[:, None]
                p_slots[r] = p_slots[r] * (1.0 - is_r) + new_pres_slot * is_r
            pres_state = tf.concat(p_slots, 1)

            # Update full state: [freq, pres, enrich, is_decoded] per round
            new_full_slot = tf.stack([act_f, act_p, act_e, tf.ones([B])], 1)  # [B, 4]
            f_slots = [full_state[:, r*FS:(r+1)*FS] for r in range(3)]
            for r in range(3):
                is_r = tf.cast(tf.equal(ridx, r), tf.float32)[:, None]
                f_slots[r] = f_slots[r] * (1.0 - is_r) + new_full_slot * is_r
            full_state = tf.concat(f_slots, 1)

            # Store decoded values in canonical order
            for r in range(3):
                is_r = tf.cast(tf.equal(ridx, r), tf.float32)
                dec_freq = dec_freq.write(r, dec_freq.read(r)*(1.-is_r) + act_f*is_r)
                dec_pres = dec_pres.write(r, dec_pres.read(r)*(1.-is_r) + act_p*is_r)
                dec_enrich = dec_enrich.write(r, dec_enrich.read(r)*(1.-is_r) + act_e*is_r)

        nm = n_masked + 1e-8
        d_f = tf.stack([dec_freq.read(r) for r in range(3)], 1)
        d_p = tf.stack([dec_pres.read(r) for r in range(3)], 1)
        d_e = tf.stack([dec_enrich.read(r) for r in range(3)], 1)
        return loss_freq/nm, loss_pres/nm, loss_enrich/nm, d_f, d_p, d_e

    def call(self, seq_embed, freq, pres, enrich, training=False):
        """Training forward: random permutation + random masking."""
        B = tf.shape(seq_embed)[0]
        perm_idx = tf.random.uniform([B], 0, 6, tf.int32)
        perms = tf.gather(ALL_PERMS, perm_idx)
        round_mask = tf.cast(tf.random.uniform([B, 3]) < self.mask_rate, tf.float32)
        lf, lp, le, df, dp, de = self._decode_steps(
            seq_embed, freq, pres, enrich, round_mask, perms, training)
        return {"ar_loss_freq": lf, "ar_loss_pres": lp, "ar_loss_enrich": le,
                "dec_freq": df, "dec_pres": dp, "dec_enrich": de}

    def predict_with_mask(self, seq_embed, freq, pres, enrich, round_mask_vec):
        """Inference: canonical order [0,1,2] with given mask."""
        B = tf.shape(seq_embed)[0]
        perms = tf.tile(tf.constant([[0,1,2]], tf.int32), [B, 1])
        _, _, _, df, dp, de = self._decode_steps(
            seq_embed, freq, pres, enrich, round_mask_vec, perms, training=False)
        return {"dec_freq": df, "dec_pres": dp, "dec_enrich": de}


# =============================================================================
# VAE (with enrichment)
# =============================================================================
class VAEEncoder(tf.keras.layers.Layer):
    """Projects [seq_embed, R0_f, R0_p, R0_e, R1_f, R1_p, R1_e, R3_f, R3_p, R3_e]
       -> hidden -> mu, logvar -> z."""
    def __init__(self, d_model, hidden_dim, latent_dim, dropout=0.1, reg=None, **kw):
        super().__init__(**kw)
        # d_model + 3*(freq+pres+enrich) = d_model + 9
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="gelu", kernel_regularizer=reg),
            tf.keras.layers.Dropout(dropout),
        ], name="vae_enc_fc")
        self.mu_layer = tf.keras.layers.Dense(latent_dim, kernel_regularizer=reg, name="mu")
        self.logvar_layer = tf.keras.layers.Dense(latent_dim, kernel_regularizer=reg, name="logvar")

    def call(self, seq_embed, dec_freq, dec_pres, dec_enrich, training=False):
        # Canonical order: [seq, R0_f, R0_p, R0_e, R1_f, R1_p, R1_e, R3_f, R3_p, R3_e]
        x = tf.concat([
            seq_embed,
            dec_freq[:, 0:1], dec_pres[:, 0:1], dec_enrich[:, 0:1],
            dec_freq[:, 1:2], dec_pres[:, 1:2], dec_enrich[:, 1:2],
            dec_freq[:, 2:3], dec_pres[:, 2:3], dec_enrich[:, 2:3],
        ], axis=1)  # [B, d_model + 9]
        h = self.fc(x, training=training)
        mu = self.mu_layer(h)
        logvar = tf.clip_by_value(self.logvar_layer(h), -LOGVAR_CLIP, LOGVAR_CLIP)
        eps = tf.random.normal(tf.shape(mu))
        z = mu + tf.exp(0.5 * logvar) * eps
        return z, mu, logvar


class VAEDecoder(tf.keras.layers.Layer):
    """Decodes z -> seq logits + round (freq, pres_logit, enrich) x 3."""
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
        # 3 rounds x (freq + pres_logit + enrich) = 9
        self.round_head = tf.keras.layers.Dense(9, kernel_regularizer=reg, name="round_recon")

    def call(self, z, training=False):
        h = self.fc(z, training=training)
        seq_logits = tf.reshape(self.seq_head(h), [-1, self.max_len, N_AA])
        rout = self.round_head(h)  # [B, 9]
        # Parse: [R0_freq, R0_pres_logit, R0_enrich, R1_..., R3_...]
        recon_freq = tf.stack([rout[:,0], rout[:,3], rout[:,6]], 1)
        recon_pres_logit = tf.stack([rout[:,1], rout[:,4], rout[:,7]], 1)
        recon_enrich = tf.stack([rout[:,2], rout[:,5], rout[:,8]], 1)
        return seq_logits, recon_freq, recon_pres_logit, recon_enrich


# =============================================================================
# FULL AUTOREG-VAE MODEL
# =============================================================================
class AutoregVAEModel(tf.keras.Model):
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
            args.d_model, args.autoreg_hidden_dim,
            mask_rate=args.autoreg_mask_rate,
            dropout=args.dropout, reg=reg, name="autoreg")
        self.vae_enc = VAEEncoder(
            args.d_model, args.vae_hidden_dim, args.vae_latent_dim,
            args.dropout, reg, name="vae_enc")
        self.vae_dec = VAEDecoder(
            args.vae_latent_dim, args.vae_hidden_dim, args.max_len,
            args.dropout, reg, name="vae_dec")
        self._log_enrich = args.log_enrichment

    def call(self, inputs, training=False):
        pooled, per_res = self.encoder(
            inputs["mlm_input"], inputs["padding_mask"], training=training)
        mlm_logits = self.mlm_head(per_res, training=training)

        enrich = compute_enrichment(inputs["freq"], self._log_enrich)
        ar_out = self.autoreg(pooled, inputs["freq"], inputs["pres"],
                              enrich, training=training)

        z, mu, logvar = self.vae_enc(
            pooled, ar_out["dec_freq"], ar_out["dec_pres"],
            ar_out["dec_enrich"], training=training)
        seq_logits, recon_freq, recon_pres_logit, recon_enrich = self.vae_dec(
            z, training=training)

        return {
            "mlm_logits": mlm_logits,
            "ar_loss_freq": ar_out["ar_loss_freq"],
            "ar_loss_pres": ar_out["ar_loss_pres"],
            "ar_loss_enrich": ar_out["ar_loss_enrich"],
            "z": z, "mu": mu, "logvar": logvar,
            "seq_logits": seq_logits,
            "recon_freq": recon_freq,
            "recon_pres_logit": recon_pres_logit,
            "recon_enrich": recon_enrich,
            "dec_freq": ar_out["dec_freq"],
            "dec_pres": ar_out["dec_pres"],
            "dec_enrich": ar_out["dec_enrich"],
        }

    def encode_to_latent(self, ids, pmask, freq, pres, round_mask_vec=None):
        pooled, _ = self.encoder(ids, pmask, training=False)
        enrich = compute_enrichment(freq, self._log_enrich)
        if round_mask_vec is None:
            round_mask_vec = tf.zeros_like(freq)
        ar = self.autoreg.predict_with_mask(pooled, freq, pres, enrich,
                                            round_mask_vec)
        z, mu, logvar = self.vae_enc(
            pooled, ar["dec_freq"], ar["dec_pres"], ar["dec_enrich"],
            training=False)
        return {"z": z, "mu": mu, "pooled": pooled,
                "dec_freq": ar["dec_freq"], "dec_pres": ar["dec_pres"],
                "dec_enrich": ar["dec_enrich"]}

    def generate_from_z(self, z):
        return self.vae_dec(z, training=False)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
def mlm_loss(logits, original_ids, mask_flag):
    labels = tf.clip_by_value(tf.cast(original_ids, tf.int32), 0, 19)
    ce = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return tf.reduce_sum(ce * mask_flag) / (tf.reduce_sum(mask_flag) + 1e-8)


def vae_recon_losses(seq_logits, original_ids, padding_mask,
                     recon_freq, true_freq,
                     recon_pres_logit, true_pres,
                     recon_enrich, true_enrich):
    """Separate VAE reconstruction losses.

    Returns:
        seq_loss: sparse CCE on sequence (masked by padding).
        freq_loss: MSE on frequency.
        pres_loss: BCE on presence.
        enrich_loss: MSE on enrichment.
    """
    labels = tf.clip_by_value(tf.cast(original_ids, tf.int32), 0, 19)
    seq_ce = tf.keras.losses.sparse_categorical_crossentropy(
        labels, seq_logits, from_logits=True)
    seq_loss = tf.reduce_sum(seq_ce * padding_mask) / (tf.reduce_sum(padding_mask) + 1e-8)
    freq_loss = tf.reduce_mean(tf.square(recon_freq - true_freq))
    pres_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=true_pres, logits=recon_pres_logit))
    enrich_loss = tf.reduce_mean(tf.square(recon_enrich - true_enrich))
    return seq_loss, freq_loss, pres_loss, enrich_loss


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
        return self.peak * tf.where(s < w, s / tf.maximum(w, 1.0),
                                    0.5*(1.+tf.cos(math.pi*(s-w)/tf.maximum(t-w,1.))))
    def get_config(self):
        return {"peak": self.peak, "warmup": self.warmup, "total": self.total}


def make_optimizer(args, total_steps):
    if args.lr_schedule == "constant": lr = args.learning_rate
    elif args.lr_schedule == "cosine":
        lr = tf.keras.optimizers.schedules.CosineDecay(args.learning_rate, total_steps, alpha=1e-6)
    else: lr = WarmupCosine(args.learning_rate, args.warmup_steps, total_steps)
    if args.optimizer == "adamw":
        return tf.keras.optimizers.AdamW(lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd": return tf.keras.optimizers.SGD(lr, momentum=0.9)
    return tf.keras.optimizers.Adam(lr)


# =============================================================================
# CHECKPOINT / STATE
# =============================================================================
def _ckpt_dir(args): return os.path.join(args.output_dir, "checkpoints")


def save_ckpt(args, stage, model, epoch, step, best_val, history):
    d = _ckpt_dir(args); os.makedirs(d, exist_ok=True)
    model.save_weights(os.path.join(d, f"{stage}_latest.weights.h5"))
    state = {"epoch": epoch, "step": step, "best_val": best_val, "history": history}
    with open(os.path.join(d, f"{stage}_state.json"), "w") as f:
        json.dump(state, f, indent=2)


def save_best(args, stage, model):
    d = _ckpt_dir(args); os.makedirs(d, exist_ok=True)
    model.save_weights(os.path.join(d, f"{stage}_best.weights.h5"))


def load_ckpt(args, stage):
    d = _ckpt_dir(args)
    sp = os.path.join(d, f"{stage}_state.json")
    wp = os.path.join(d, f"{stage}_latest.weights.h5")
    if os.path.exists(sp) and os.path.exists(wp):
        with open(sp) as f: s = json.load(f)
        s["weights_path"] = wp; return s
    return None


def _init_model(cls, args, dummy):
    m = cls(args); _ = m(dummy, training=False); return m


def _mlm_dummy(args):
    return {"input_ids": tf.zeros([2, args.max_len], tf.int32),
            "padding_mask": tf.ones([2, args.max_len], tf.float32)}


def _autoreg_dummy(args):
    return {"seq": tf.zeros([2, args.max_len], tf.int32),
            "mlm_input": tf.zeros([2, args.max_len], tf.int32),
            "mlm_mask": tf.zeros([2, args.max_len], tf.float32),
            "padding_mask": tf.ones([2, args.max_len], tf.float32),
            "freq": tf.ones([2, 3], tf.float32) * 0.01,
            "pres": tf.ones([2, 3], tf.float32)}


def _set_trainable(layer, val): layer.trainable = val


def _save_component_weights(layer, path):
    ws = layer.get_weights()
    np.savez(path, *ws, _count=len(ws))


def _load_component_weights(layer, path):
    data = np.load(path, allow_pickle=True)
    ws = [data[f"arr_{i}"] for i in range(int(data["_count"]))]
    layer.set_weights(ws)


def _save_encoder_only(args, model):
    cd = _ckpt_dir(args); os.makedirs(cd, exist_ok=True)
    for name, layer in [("encoder", model.encoder), ("mlm_head", model.mlm_head),
                        ("autoreg", model.autoreg), ("vae_enc", model.vae_enc),
                        ("vae_dec", model.vae_dec)]:
        _save_component_weights(layer, os.path.join(cd, f"component_{name}.npz"))
    log.info("Saved component weights.")


def _save_generator(args, model):
    cd = _ckpt_dir(args); os.makedirs(cd, exist_ok=True)
    _save_component_weights(model.vae_dec, os.path.join(cd, "generator.npz"))
    log.info("Saved generator (VAE decoder) weights.")


# =============================================================================
# TRAINING: MLM
# =============================================================================
def train_mlm(args, meta):
    log.info("=" * 60); log.info("STAGE: MLM PRETRAINING (R0 only)"); log.info("=" * 60)
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
    opt = make_optimizer(args, total); gc = args.grad_clip

    loss_acc = tf.Variable(0.0, trainable=False)
    step_acc = tf.Variable(0, trainable=False, dtype=tf.int32)
    acc1_acc = tf.Variable(0.0, trainable=False)
    acc5_acc = tf.Variable(0.0, trainable=False)
    nmask_acc = tf.Variable(0.0, trainable=False)

    @tf.function(jit_compile=args.xla)
    def train_step(batch):
        with tf.GradientTape() as tape:
            logits = model(batch, training=True)
            loss = mlm_loss(logits, batch["original_ids"], batch["mask_flag"])
            loss += sum(model.losses)
        gs = tape.gradient(loss, model.trainable_variables)
        gs, _ = tf.clip_by_global_norm(gs, gc)
        opt.apply_gradients(zip(gs, model.trainable_variables))
        labels = tf.clip_by_value(tf.cast(batch["original_ids"], tf.int32), 0, 19)
        mf = batch["mask_flag"]; nm = tf.reduce_sum(mf)
        pred1 = tf.argmax(logits, axis=-1, output_type=tf.int32)
        c1 = tf.reduce_sum(tf.cast(tf.equal(pred1, labels), tf.float32) * mf)
        _, top5 = tf.math.top_k(logits, k=5)
        hit5 = tf.reduce_any(tf.equal(top5, labels[:,:,tf.newaxis]), axis=-1)
        c5 = tf.reduce_sum(tf.cast(hit5, tf.float32) * mf)
        loss_acc.assign_add(loss); step_acc.assign_add(1)
        acc1_acc.assign_add(c1); acc5_acc.assign_add(c5); nmask_acc.assign_add(nm)

    @tf.function(jit_compile=args.xla)
    def val_step(batch):
        logits = model(batch, training=False)
        loss = mlm_loss(logits, batch["original_ids"], batch["mask_flag"])
        labels = tf.clip_by_value(tf.cast(batch["original_ids"], tf.int32), 0, 19)
        mf = batch["mask_flag"]; nm = tf.reduce_sum(mf)
        pred1 = tf.argmax(logits, axis=-1, output_type=tf.int32)
        c1 = tf.reduce_sum(tf.cast(tf.equal(pred1, labels), tf.float32) * mf)
        _, top5 = tf.math.top_k(logits, k=5)
        hit5 = tf.reduce_any(tf.equal(top5, labels[:,:,tf.newaxis]), axis=-1)
        c5 = tf.reduce_sum(tf.cast(hit5, tf.float32) * mf)
        return loss, c1, c5, nm, logits

    start_ep, gstep, best_val, patience_ctr = 1, 0, float("inf"), 0
    history = {"train_loss":[],"val_loss":[],"train_ppl":[],"val_ppl":[],
               "train_top1":[],"val_top1":[],"train_top5":[],"val_top5":[]}
    if args.resume:
        st = load_ckpt(args, "mlm")
        if st:
            model.load_weights(st["weights_path"])
            start_ep, gstep, best_val = st["epoch"]+1, st["step"], st["best_val"]
            history = st.get("history", history)
            for k in history: history.setdefault(k, [])
            log.info("Resumed MLM: ep=%d step=%d best=%.4f", start_ep-1, gstep, best_val)

    for ep in range(start_ep, args.epochs + 1):
        t0 = time.time()
        for v in [loss_acc, acc1_acc, acc5_acc, nmask_acc]: v.assign(0.0)
        step_acc.assign(0)
        for batch in train_ds.take(spe):
            train_step(batch); gstep += 1
            if gstep % args.log_every == 0:
                s = max(int(step_acc), 1); nm = max(float(nmask_acc), 1.)
                log.info("  step %d ep %d | loss=%.4f top1=%.3f top5=%.3f (%.0f s/s)",
                         gstep, ep, float(loss_acc)/s, float(acc1_acc)/nm,
                         float(acc5_acc)/nm, s*args.batch_size/(time.time()-t0))
            if args.ckpt_every > 0 and gstep % args.ckpt_every == 0:
                save_ckpt(args, "mlm", model, ep, gstep, best_val, history)

        tl = float(loss_acc)/max(int(step_acc),1)
        t_nm = max(float(nmask_acc),1.); t1 = float(acc1_acc)/t_nm; t5 = float(acc5_acc)/t_nm
        t_ppl = math.exp(min(tl, 30.))

        vl_s, v1, v5, vnm, vc = 0., 0., 0., 0., 0
        viz_batch, viz_logits = None, None
        for b in val_ds.take(vs):
            l, c1, c5, nm, logits_v = val_step(b); vc += 1
            vl_s += float(l); v1 += float(c1); v5 += float(c5); vnm += float(nm)
            if viz_batch is None:
                viz_batch = {k: v.numpy() for k, v in b.items()}
                viz_logits = logits_v.numpy()
        vl = vl_s/max(vc,1); vnm = max(vnm,1.)
        v_t1, v_t5 = v1/vnm, v5/vnm; v_ppl = math.exp(min(vl, 30.))
        elapsed = time.time()-t0
        log.info("Epoch %d/%d (%.1fs) | TRAIN loss=%.4f ppl=%.2f top1=%.3f top5=%.3f"
                 " | VAL loss=%.4f ppl=%.2f top1=%.3f top5=%.3f",
                 ep, args.epochs, elapsed, tl, t_ppl, t1, t5, vl, v_ppl, v_t1, v_t5)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_ppl"].append(t_ppl); history["val_ppl"].append(v_ppl)
        history["train_top1"].append(t1); history["val_top1"].append(v_t1)
        history["train_top5"].append(t5); history["val_top5"].append(v_t5)

        if vl < best_val - args.min_delta:
            best_val, patience_ctr = vl, 0
            save_best(args, "mlm", model); log.info("  -> new best MLM %.4f", vl)
        else: patience_ctr += 1
        save_ckpt(args, "mlm", model, ep, gstep, best_val, history)
        if args.convergence and patience_ctr >= args.patience:
            log.info("Early stop at epoch %d", ep); break

    model.save_weights(os.path.join(_ckpt_dir(args), "mlm_final.weights.h5"))
    if viz_batch is not None: _save_mlm_seq_viz(args, viz_batch, viz_logits)
    log.info("MLM done. Best val=%.4f", best_val)
    return model


def _save_mlm_seq_viz(args, batch, logits, n_show=10):
    idx2aa = {i: a for i, a in enumerate(AA)}
    idx2aa[PAD] = "."; idx2aa[MASK] = "_"; idx2aa[UNK] = "X"
    out_dir = os.path.join(args.output_dir, "analysis"); os.makedirs(out_dir, exist_ok=True)
    orig, inp, mflag, pmask = batch["original_ids"], batch["input_ids"], batch["mask_flag"], batch["padding_mask"]
    pred = np.argmax(logits, axis=-1)
    B = orig.shape[0]; indices = np.random.choice(B, min(n_show, B), replace=False)
    lines = ["="*80, "MLM Masked-Token Reconstruction Visualization",
             f"Showing {len(indices)} random sequences",
             "Legend: UPPER=masked, lower=unmasked, .=pad", "="*80, ""]
    for ci, i in enumerate(indices):
        slen = int(np.sum(pmask[i]))
        os_, ps_, is_, ms_ = [], [], [], []
        nc, nm = 0, 0
        for j in range(slen):
            ao, ap = idx2aa.get(int(orig[i,j]),"X"), idx2aa.get(int(pred[i,j]),"X")
            im = mflag[i,j] > 0.5
            if im:
                nm += 1; os_.append(ao.upper()); ps_.append(ap.upper()); is_.append("_")
                ms_.append("^" if ao==ap else "X")
                if ao==ap: nc += 1
            else:
                os_.append(ao.lower()); ps_.append(ap.lower()); is_.append(ao.lower()); ms_.append(" ")
        acc = nc/max(nm,1)
        lines += [f"--- Seq {ci+1} (len={slen} masked={nm} acc={acc:.2%}) ---",
                  f"  Input:    {''.join(is_)}", f"  Original: {''.join(os_)}",
                  f"  Predict:  {''.join(ps_)}", f"  Match:    {''.join(ms_)}",
                  f"            (^ correct, X wrong)", ""]
    with open(os.path.join(out_dir, "mlm_seq_visualization.txt"), "w") as f:
        f.write("\n".join(lines))
    log.info("Saved MLM sequence visualization")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        n = len(indices)
        fig, axes = plt.subplots(n, 1, figsize=(min(24, 0.18*120+2), n*1.8+1.5))
        if n == 1: axes = [axes]
        fig.suptitle("MLM Masked-Token Reconstruction", fontsize=14, fontweight="bold")
        for ai, idx in enumerate(indices):
            ax = axes[ai]; slen = min(int(np.sum(pmask[idx])), 120)
            ax.set_xlim(-0.5, slen+0.5); ax.set_ylim(-0.5, 2.5); ax.axis("off")
            nc_l, nm_l = 0, 0
            for j in range(slen):
                ao, ap = idx2aa.get(int(orig[idx,j]),"X"), idx2aa.get(int(pred[idx,j]),"X")
                im = mflag[idx,j]>0.5
                if im: nm_l += 1; nc_l += int(ao==ap)
                co = "#FFD700" if im else "#E8E8E8"
                ax.add_patch(FancyBboxPatch((j-.4,1.1),.8,.8,boxstyle="round,pad=0.05",fc=co,ec="gray",lw=.5))
                ax.text(j,1.5,ao,ha="center",va="center",fontsize=6,fontfamily="monospace",fontweight="bold" if im else "normal")
                cp = ("#90EE90" if ao==ap else "#FF6B6B") if im else "#E8E8E8"
                ax.add_patch(FancyBboxPatch((j-.4,0),.8,.8,boxstyle="round,pad=0.05",fc=cp,ec="gray",lw=.5))
                ax.text(j,.4,ap,ha="center",va="center",fontsize=6,fontfamily="monospace",fontweight="bold" if im else "normal")
            la = nc_l/max(nm_l,1)
            ax.text(-.3,2.2,f"Seq {ai+1} len={slen} masked={nm_l} acc={la:.0%}",fontsize=7,fontweight="bold")
            ax.text(slen+.5,1.5,"orig",fontsize=6,va="center"); ax.text(slen+.5,.4,"pred",fontsize=6,va="center")
        fig.text(.01,.01,"Gold=masked | Green=correct | Red=incorrect",fontsize=7,style="italic")
        fig.tight_layout(rect=[0,.03,1,.96])
        fig.savefig(os.path.join(out_dir,"mlm_seq_visualization.png"),dpi=200,bbox_inches="tight"); plt.close(fig)
        log.info("Saved MLM sequence figure")
    except ImportError: log.warning("matplotlib not available — skipped figure.")


# =============================================================================
# TRAINING: AUTOREGRESSIVE VAE
# =============================================================================
def train_autoreg_vae(args, meta):
    log.info("=" * 60); log.info("STAGE: AUTOREGRESSIVE VAE"); log.info("=" * 60)
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

    # Load pretrained MLM encoder
    cd = _ckpt_dir(args)
    mlm_path = None
    for wn in ["mlm_best.weights.h5", "mlm_final.weights.h5"]:
        wp = os.path.join(cd, wn)
        if os.path.exists(wp): mlm_path = wp; break
    if mlm_path:
        log.info("Loading MLM encoder from: %s", mlm_path)
        mlm_tmp = _init_model(MLMModel, args, _mlm_dummy(args))
        mlm_tmp.load_weights(mlm_path)
        for mv, ev in zip(model.encoder.trainable_variables, mlm_tmp.encoder.trainable_variables):
            mv.assign(ev)
        for mv, ev in zip(model.mlm_head.trainable_variables, mlm_tmp.mlm_head.trainable_variables):
            mv.assign(ev)
        log.info("  -> transferred encoder + mlm_head weights"); del mlm_tmp
    else:
        log.warning("No MLM weights found. Training from scratch.")

    if args.freeze_mlm:
        _set_trainable(model.encoder, False); _set_trainable(model.mlm_head, False)
        log.info("Frozen: encoder + mlm_head")
    if args.freeze_autoreg:
        _set_trainable(model.autoreg, False); log.info("Frozen: autoreg")
    if args.freeze_vae:
        _set_trainable(model.vae_enc, False); _set_trainable(model.vae_dec, False)
        log.info("Frozen: vae_enc + vae_dec")

    opt = make_optimizer(args, total); gc = args.grad_clip
    w_rs, w_rr, w_ar, w_kl = args.w_recon_seq, args.w_recon_round, args.w_autoreg, args.kl_weight

    # GPU-resident accumulators — separate for each loss component
    a_tot = tf.Variable(0.0, trainable=False)
    a_mlm = tf.Variable(0.0, trainable=False)
    a_ar_freq = tf.Variable(0.0, trainable=False)
    a_ar_pres = tf.Variable(0.0, trainable=False)
    a_ar_enrich = tf.Variable(0.0, trainable=False)
    a_kl = tf.Variable(0.0, trainable=False)
    a_vae_seq = tf.Variable(0.0, trainable=False)
    a_vae_freq = tf.Variable(0.0, trainable=False)
    a_vae_pres = tf.Variable(0.0, trainable=False)
    a_vae_enrich = tf.Variable(0.0, trainable=False)
    a_st = tf.Variable(0, trainable=False, dtype=tf.int32)

    mf_weight = args.mlm_mask_frac  # weight for aux MLM loss
    use_log_e = args.log_enrichment

    @tf.function(jit_compile=args.xla)
    def train_step(batch):
        with tf.GradientTape() as tape:
            out = model(batch, training=True)
            enrich_gt = compute_enrichment(batch["freq"], use_log_e)
            # MLM auxiliary
            lm = mlm_loss(out["mlm_logits"], batch["seq"], batch["mlm_mask"]) * mf_weight
            # Autoreg losses (already averaged over masked positions)
            l_ar_f = out["ar_loss_freq"] * w_ar
            l_ar_p = out["ar_loss_pres"] * w_ar
            l_ar_e = out["ar_loss_enrich"] * w_ar
            # VAE reconstruction
            vs, vf, vp, ve = vae_recon_losses(
                out["seq_logits"], batch["seq"], batch["padding_mask"],
                out["recon_freq"], batch["freq"],
                out["recon_pres_logit"], batch["pres"],
                out["recon_enrich"], enrich_gt)
            l_vae_seq = vs * w_rs
            l_vae_freq = vf * w_rr
            l_vae_pres = vp * w_rr
            l_vae_enrich = ve * w_rr
            # KL
            lk = kl_divergence(out["mu"], out["logvar"]) * w_kl
            total_loss = (lm + l_ar_f + l_ar_p + l_ar_e +
                          l_vae_seq + l_vae_freq + l_vae_pres + l_vae_enrich +
                          lk + sum(model.losses))
        gs = tape.gradient(total_loss, model.trainable_variables)
        gs, _ = tf.clip_by_global_norm(gs, gc)
        opt.apply_gradients(zip(gs, model.trainable_variables))
        a_tot.assign_add(total_loss); a_mlm.assign_add(lm)
        a_ar_freq.assign_add(l_ar_f); a_ar_pres.assign_add(l_ar_p)
        a_ar_enrich.assign_add(l_ar_e); a_kl.assign_add(lk)
        a_vae_seq.assign_add(l_vae_seq); a_vae_freq.assign_add(l_vae_freq)
        a_vae_pres.assign_add(l_vae_pres); a_vae_enrich.assign_add(l_vae_enrich)
        a_st.assign_add(1)

    @tf.function(jit_compile=args.xla)
    def val_step(batch):
        out = model(batch, training=False)
        enrich_gt = compute_enrichment(batch["freq"], use_log_e)
        vs, vf, vp, ve = vae_recon_losses(
            out["seq_logits"], batch["seq"], batch["padding_mask"],
            out["recon_freq"], batch["freq"],
            out["recon_pres_logit"], batch["pres"],
            out["recon_enrich"], enrich_gt)
        lk = kl_divergence(out["mu"], out["logvar"])
        return ((out["ar_loss_freq"]+out["ar_loss_pres"]+out["ar_loss_enrich"])*w_ar +
                vs*w_rs + (vf+vp+ve)*w_rr + lk*w_kl)

    # Resume
    start_ep, gstep, best_val, patience_ctr = 1, 0, float("inf"), 0
    history = {
        "train_loss": [], "val_loss": [],
        "mlm": [],
        "ar_freq": [], "ar_pres": [], "ar_enrich": [],
        "kl": [],
        "vae_seq_recon": [], "vae_freq_recon": [], "vae_pres_recon": [], "vae_enrich_recon": [],
    }
    if args.resume:
        st = load_ckpt(args, "autoreg_vae")
        if st:
            model.load_weights(st["weights_path"])
            start_ep, gstep, best_val = st["epoch"]+1, st["step"], st["best_val"]
            history = st.get("history", history)
            for k in history: history.setdefault(k, [])
            log.info("Resumed autoreg_vae: ep=%d step=%d best=%.4f", start_ep-1, gstep, best_val)

    for ep in range(start_ep, args.epochs + 1):
        t0 = time.time()
        for v in [a_tot, a_mlm, a_ar_freq, a_ar_pres, a_ar_enrich,
                  a_kl, a_vae_seq, a_vae_freq, a_vae_pres, a_vae_enrich]:
            v.assign(0.0)
        a_st.assign(0)

        for batch in train_ds.take(spe):
            train_step(batch); gstep += 1
            if gstep % args.log_every == 0:
                s = max(int(a_st), 1)
                log.info("  step %d ep %d | tot=%.4f mlm=%.4f "
                         "ar_f=%.4f ar_p=%.4f ar_e=%.4f kl=%.4f "
                         "vs=%.4f vf=%.4f vp=%.4f ve=%.4f (%.0f s/s)",
                         gstep, ep,
                         float(a_tot)/s, float(a_mlm)/s,
                         float(a_ar_freq)/s, float(a_ar_pres)/s, float(a_ar_enrich)/s,
                         float(a_kl)/s,
                         float(a_vae_seq)/s, float(a_vae_freq)/s,
                         float(a_vae_pres)/s, float(a_vae_enrich)/s,
                         s*args.batch_size/(time.time()-t0))
            if args.ckpt_every > 0 and gstep % args.ckpt_every == 0:
                save_ckpt(args, "autoreg_vae", model, ep, gstep, best_val, history)

        s = max(int(a_st), 1)
        tl = float(a_tot)/s
        vl_s, vc = 0.0, 0
        for b in val_ds.take(vs):
            vl_s += float(val_step(b)); vc += 1
        vl = vl_s / max(vc, 1)
        elapsed = time.time() - t0
        log.info("Epoch %d/%d train=%.4f val=%.4f (%.1fs) | "
                 "mlm=%.4f ar_f=%.4f ar_p=%.4f ar_e=%.4f kl=%.4f "
                 "vs=%.4f vf=%.4f vp=%.4f ve=%.4f",
                 ep, args.epochs, tl, vl, elapsed,
                 float(a_mlm)/s, float(a_ar_freq)/s, float(a_ar_pres)/s,
                 float(a_ar_enrich)/s, float(a_kl)/s,
                 float(a_vae_seq)/s, float(a_vae_freq)/s,
                 float(a_vae_pres)/s, float(a_vae_enrich)/s)

        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["mlm"].append(float(a_mlm)/s)
        history["ar_freq"].append(float(a_ar_freq)/s)
        history["ar_pres"].append(float(a_ar_pres)/s)
        history["ar_enrich"].append(float(a_ar_enrich)/s)
        history["kl"].append(float(a_kl)/s)
        history["vae_seq_recon"].append(float(a_vae_seq)/s)
        history["vae_freq_recon"].append(float(a_vae_freq)/s)
        history["vae_pres_recon"].append(float(a_vae_pres)/s)
        history["vae_enrich_recon"].append(float(a_vae_enrich)/s)

        if vl < best_val - args.min_delta:
            best_val, patience_ctr = vl, 0
            save_best(args, "autoreg_vae", model); log.info("  -> new best %.4f", vl)
        else: patience_ctr += 1
        save_ckpt(args, "autoreg_vae", model, ep, gstep, best_val, history)
        if args.convergence and patience_ctr >= args.patience:
            log.info("Early stop at epoch %d", ep); break

    model.save_weights(os.path.join(_ckpt_dir(args), "autoreg_vae_final.weights.h5"))
    _save_encoder_only(args, model)
    _save_generator(args, model)
    log.info("Autoreg-VAE done. Best val=%.4f", best_val)
    return model


# =============================================================================
# INFERENCE
# =============================================================================
def run_inference(args, meta):
    log.info("=" * 60); log.info("INFERENCE"); log.info("=" * 60)
    model = _init_model(AutoregVAEModel, args, _autoreg_dummy(args))
    cd = _ckpt_dir(args)
    loaded = False
    for wn in ["autoreg_vae_best.weights.h5", "autoreg_vae_final.weights.h5",
               "autoreg_vae_latest.weights.h5"]:
        wp = os.path.join(cd, wn)
        if os.path.exists(wp):
            model.load_weights(wp); log.info("Loaded: %s", wp); loaded = True; break
    if not loaded: log.error("No autoreg_vae weights in %s", cd); return

    mask_rounds = set()
    if args.inference_mask_rounds:
        for r in args.inference_mask_rounds.split(","):
            r = r.strip().upper()
            if r in ROUND_NAMES: mask_rounds.add(ROUND_NAMES.index(r))
    log.info("Masking rounds: %s", [ROUND_NAMES[i] for i in mask_rounds])

    out_dir = os.path.join(args.output_dir, "inference"); os.makedirs(out_dir, exist_ok=True)
    results = {k: [] for k in ["z","mu","dec_freq","dec_pres","dec_enrich","pooled"]}
    total = 0
    for prefix in ["train", "val", "test"]:
        tfdir = os.path.join(args.output_dir, "tfrecords")
        if not glob.glob(os.path.join(tfdir, f"{prefix}_*.tfrecord")): continue
        ds = build_dataset(args, prefix, "inference", shuffle=False)
        for batch in ds:
            B = tf.shape(batch["input_ids"])[0]
            rmask = tf.zeros([B, 3])
            if mask_rounds:
                cols = [tf.ones([B]) if r in mask_rounds else tf.zeros([B]) for r in range(3)]
                rmask = tf.stack(cols, 1)
            out = model.encode_to_latent(batch["input_ids"], batch["padding_mask"],
                                         batch["freq"], batch["pres"], rmask)
            for k in results: results[k].append(out[k].numpy())
            total += int(B)
            if total % (args.inference_batch_size*100) < args.inference_batch_size:
                log.info("  processed %d", total)
    for k in results: results[k] = np.concatenate(results[k], 0)
    N = results["z"].shape[0]; log.info("Total: %d sequences", N)
    np.savez_compressed(os.path.join(out_dir, "latents.npz"), **results)
    ids_p = os.path.join(args.output_dir, "tfrecords", "ids_train.npy")
    ids = np.load(ids_p, allow_pickle=True) if os.path.exists(ids_p) else None
    with open(os.path.join(out_dir, "predictions.tsv"), "w") as f:
        f.write("seq_id\tz_norm\tdec_freq_R0\tdec_freq_R1\tdec_freq_R3\t"
                "dec_pres_R0\tdec_pres_R1\tdec_pres_R3\t"
                "dec_enrich_R0\tdec_enrich_R1\tdec_enrich_R3\n")
        for i in range(N):
            sid = ids[i] if ids is not None and i < len(ids) else str(i)
            zn = np.linalg.norm(results["z"][i])
            f.write(f"{sid}\t{zn:.6f}\t"
                    f"{results['dec_freq'][i,0]:.6f}\t{results['dec_freq'][i,1]:.6f}\t"
                    f"{results['dec_freq'][i,2]:.6f}\t"
                    f"{results['dec_pres'][i,0]:.4f}\t{results['dec_pres'][i,1]:.4f}\t"
                    f"{results['dec_pres'][i,2]:.4f}\t"
                    f"{results['dec_enrich'][i,0]:.4f}\t{results['dec_enrich'][i,1]:.4f}\t"
                    f"{results['dec_enrich'][i,2]:.4f}\n")
    log.info("Saved inference results to %s", out_dir)


# =============================================================================
# GENERATOR
# =============================================================================
def run_generator(args, meta):
    log.info("=" * 60); log.info("GENERATOR"); log.info("=" * 60)
    model = _init_model(AutoregVAEModel, args, _autoreg_dummy(args))
    cd = _ckpt_dir(args)
    for wn in ["autoreg_vae_best.weights.h5", "autoreg_vae_final.weights.h5"]:
        wp = os.path.join(cd, wn)
        if os.path.exists(wp): model.load_weights(wp); log.info("Loaded: %s", wp); break
    else: log.error("No weights found"); return

    out_dir = os.path.join(args.output_dir, "generated"); os.makedirs(out_dir, exist_ok=True)
    n = args.n_generate
    z = tf.random.normal([n, args.vae_latent_dim])
    seq_logits, recon_freq, recon_pres_logit, recon_enrich = model.generate_from_z(z)
    seq_ids = tf.argmax(seq_logits, -1).numpy()
    rf, rp, re = recon_freq.numpy(), tf.nn.sigmoid(recon_pres_logit).numpy(), recon_enrich.numpy()
    idx2aa = {i: a for a, i in AA2I.items()}
    with open(os.path.join(out_dir, "generated.fasta"), "w") as ff:
        with open(os.path.join(out_dir, "generated_rounds.tsv"), "w") as ft:
            ft.write("gen_id\tfreq_R0\tfreq_R1\tfreq_R3\tpres_R0\tpres_R1\tpres_R3\t"
                     "enrich_R0\tenrich_R1\tenrich_R3\n")
            for i in range(n):
                seq = "".join(idx2aa.get(int(aa), "X") for aa in seq_ids[i] if int(aa) < N_AA)
                ff.write(f">gen_{i}\n{seq}\n")
                ft.write(f"gen_{i}\t{rf[i,0]:.6f}\t{rf[i,1]:.6f}\t{rf[i,2]:.6f}\t"
                         f"{rp[i,0]:.4f}\t{rp[i,1]:.4f}\t{rp[i,2]:.4f}\t"
                         f"{re[i,0]:.4f}\t{re[i,1]:.4f}\t{re[i,2]:.4f}\n")
    log.info("Generated %d sequences -> %s", n, out_dir)


# =============================================================================
# ANALYSIS
# =============================================================================
def run_analysis(args, meta):
    log.info("=" * 60); log.info("ANALYSIS"); log.info("=" * 60)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        log.error("matplotlib required for --analysis"); return

    out_dir = os.path.join(args.output_dir, "analysis"); os.makedirs(out_dir, exist_ok=True)

    # ========== MLM curves ==========
    mlm_sp = os.path.join(_ckpt_dir(args), "mlm_state.json")
    if os.path.exists(mlm_sp):
        with open(mlm_sp) as f: hist = json.load(f).get("history", {})

        # Individual MLM plots
        mlm_panels = [
            ("train_loss", "val_loss", "MLM CE Loss", "Loss", "mlm_loss.png"),
            ("train_ppl", "val_ppl", "MLM Perplexity (exp(CE))", "PPL", "mlm_perplexity.png"),
            ("train_top1", "val_top1", "MLM Top-1 Masked Accuracy", "Accuracy", "mlm_top1.png"),
            ("train_top5", "val_top5", "MLM Top-5 Masked Accuracy", "Accuracy", "mlm_top5.png"),
        ]
        for tk, vk, title, ylabel, fname in mlm_panels:
            if not hist.get(tk): continue
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(hist[tk], label="train", lw=1.5)
            if hist.get(vk): ax.plot(hist[vk], label="val", lw=1.5)
            ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
            ax.legend(); ax.grid(alpha=0.3)
            if "Acc" in ylabel: ax.set_ylim(0, 1)
            fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
            plt.close(fig); log.info("Saved %s", fname)

        # 4-panel MLM summary
        if all(hist.get(k) for k in ["train_loss","train_ppl","train_top1","train_top5"]):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for ax, (tk, vk, title, ylabel, _) in zip(axes.flat, mlm_panels):
                ax.plot(hist[tk], label="train", lw=1.5)
                if hist.get(vk): ax.plot(hist[vk], label="val", lw=1.5)
                ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
                ax.legend(); ax.grid(alpha=0.3)
                if "Acc" in ylabel: ax.set_ylim(0, 1)
            fig.suptitle("MLM Training Summary", fontsize=14, fontweight="bold")
            fig.tight_layout(rect=[0,0,1,.96])
            fig.savefig(os.path.join(out_dir, "mlm_summary.png"), dpi=150, bbox_inches="tight")
            plt.close(fig); log.info("Saved mlm_summary.png")

    # MLM sequence visualization from model
    mlm_wt = None
    for wn in ["mlm_best.weights.h5", "mlm_final.weights.h5"]:
        wp = os.path.join(_ckpt_dir(args), wn)
        if os.path.exists(wp): mlm_wt = wp; break
    if mlm_wt:
        log.info("Loading MLM for sequence visualization...")
        mlm_m = _init_model(MLMModel, args, _mlm_dummy(args))
        mlm_m.load_weights(mlm_wt)
        val_ds = build_dataset(args, "val", "mlm", shuffle=False)
        for b in val_ds.take(1):
            lo = mlm_m(b, training=False).numpy()
            _save_mlm_seq_viz(args, {k: v.numpy() for k,v in b.items()}, lo, 10); break
        del mlm_m

    # ========== Autoreg-VAE curves ==========
    av_sp = os.path.join(_ckpt_dir(args), "autoreg_vae_state.json")
    if os.path.exists(av_sp):
        with open(av_sp) as f: hist = json.load(f).get("history", {})

        # Total loss
        if hist.get("train_loss"):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(hist["train_loss"], label="train", lw=1.5)
            if hist.get("val_loss"): ax.plot(hist["val_loss"], label="val", lw=1.5)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.set_title("Autoreg-VAE Total Loss"); ax.legend(); ax.grid(alpha=0.3)
            fig.savefig(os.path.join(out_dir, "av_total_loss.png"), dpi=150, bbox_inches="tight")
            plt.close(fig); log.info("Saved av_total_loss.png")

        # Individual component plots — each in its own figure (different scales)
        comp_plots = [
            ("kl", "KL Divergence", "KL", "av_kl.png"),
            ("vae_seq_recon", "VAE Sequence Reconstruction (CCE)", "Loss", "av_vae_seq_recon.png"),
            ("vae_freq_recon", "VAE Frequency Reconstruction (MSE)", "Loss", "av_vae_freq_recon.png"),
            ("vae_pres_recon", "VAE Presence Reconstruction (BCE)", "Loss", "av_vae_pres_recon.png"),
            ("vae_enrich_recon", "VAE Enrichment Reconstruction (MSE)", "Loss", "av_vae_enrich_recon.png"),
            ("ar_freq", "Autoreg Frequency Prediction (MSE)", "Loss", "av_ar_freq.png"),
            ("ar_pres", "Autoreg Presence Prediction (BCE)", "Loss", "av_ar_pres.png"),
            ("ar_enrich", "Autoreg Enrichment Prediction (MSE)", "Loss", "av_ar_enrich.png"),
            ("mlm", "Auxiliary MLM Loss", "Loss", "av_mlm_aux.png"),
        ]
        for key, title, ylabel, fname in comp_plots:
            if not hist.get(key): continue
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(hist[key], lw=1.5, color="tab:blue")
            ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
            ax.grid(alpha=0.3)
            fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
            plt.close(fig); log.info("Saved %s", fname)

        # Grouped: autoreg components in one figure
        ar_keys = ["ar_freq", "ar_pres", "ar_enrich"]
        if all(hist.get(k) for k in ar_keys):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for ax, k, title in zip(axes, ar_keys, ["Freq (MSE)","Presence (BCE)","Enrichment (MSE)"]):
                ax.plot(hist[k], lw=1.5); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
                ax.set_title(f"Autoreg {title}"); ax.grid(alpha=0.3)
            fig.suptitle("Autoregressive Component Losses", fontsize=14, fontweight="bold")
            fig.tight_layout(rect=[0,0,1,.94])
            fig.savefig(os.path.join(out_dir, "av_autoreg_summary.png"), dpi=150, bbox_inches="tight")
            plt.close(fig); log.info("Saved av_autoreg_summary.png")

        # Grouped: VAE recon components
        vae_keys = ["vae_seq_recon", "vae_freq_recon", "vae_pres_recon", "vae_enrich_recon"]
        if all(hist.get(k) for k in vae_keys):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for ax, k, title in zip(axes.flat, vae_keys,
                                    ["Seq (CCE)","Freq (MSE)","Presence (BCE)","Enrichment (MSE)"]):
                ax.plot(hist[k], lw=1.5); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
                ax.set_title(f"VAE Recon: {title}"); ax.grid(alpha=0.3)
            fig.suptitle("VAE Reconstruction Losses", fontsize=14, fontweight="bold")
            fig.tight_layout(rect=[0,0,1,.95])
            fig.savefig(os.path.join(out_dir, "av_vae_recon_summary.png"), dpi=150, bbox_inches="tight")
            plt.close(fig); log.info("Saved av_vae_recon_summary.png")

    # ========== PR / AUC ROC on predicted presence per round ==========
    lat_path = os.path.join(args.output_dir, "inference", "latents.npz")
    if not os.path.exists(lat_path):
        cd = _ckpt_dir(args)
        has_model = any(os.path.exists(os.path.join(cd, w))
                        for w in ["autoreg_vae_best.weights.h5", "autoreg_vae_final.weights.h5"])
        if has_model:
            log.info("Running inference for presence AUC/PR + UMAP...")
            run_inference(args, meta)

    _has_sklearn = False
    if os.path.exists(lat_path):
        try:
            from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
            _has_sklearn = True
        except ImportError:
            log.warning("scikit-learn not installed — skipping AUC/PR.")

    if os.path.exists(lat_path) and _has_sklearn:
        lat = np.load(lat_path)
        pred_pres = lat["dec_pres"]  # [N, 3] predicted presence from autoreg

        # Collect ground-truth presence from TFRecords
        gt_pres_list = []
        for prefix in ["train", "val", "test"]:
            tfdir = os.path.join(args.output_dir, "tfrecords")
            if not glob.glob(os.path.join(tfdir, f"{prefix}_*.tfrecord")): continue
            ds = build_dataset(args, prefix, "inference", shuffle=False)
            for batch in ds:
                gt_pres_list.append(batch["pres"].numpy())
        if gt_pres_list:
            gt_pres = np.concatenate(gt_pres_list, 0)
            n_samples = min(len(gt_pres), len(pred_pres))
            gt_pres, pred_pres = gt_pres[:n_samples], pred_pres[:n_samples]

            # Per-round ROC and PR
            fig_roc, axes_roc = plt.subplots(1, 3, figsize=(18, 5))
            fig_pr, axes_pr = plt.subplots(1, 3, figsize=(18, 5))
            for ri, rname in enumerate(ROUND_NAMES):
                yt, ys = gt_pres[:, ri], pred_pres[:, ri]
                if len(np.unique(yt)) < 2:
                    log.info("  %s presence: single class — skipping", rname)
                    axes_roc[ri].set_title(f"{rname} — single class")
                    axes_pr[ri].set_title(f"{rname} — single class")
                    continue
                auc = roc_auc_score(yt, ys)
                ap = average_precision_score(yt, ys)
                log.info("  %s presence: AUC=%.4f  AP=%.4f", rname, auc, ap)

                fpr, tpr, _ = roc_curve(yt, ys)
                axes_roc[ri].plot(fpr, tpr, lw=1.5)
                axes_roc[ri].plot([0,1],[0,1],"--",color="gray",alpha=0.5)
                axes_roc[ri].set_xlabel("FPR"); axes_roc[ri].set_ylabel("TPR")
                axes_roc[ri].set_title(f"{rname} Presence ROC (AUC={auc:.3f})")
                axes_roc[ri].grid(alpha=0.3)

                prec, rec, _ = precision_recall_curve(yt, ys)
                axes_pr[ri].plot(rec, prec, lw=1.5)
                axes_pr[ri].set_xlabel("Recall"); axes_pr[ri].set_ylabel("Precision")
                axes_pr[ri].set_title(f"{rname} Presence PR (AP={ap:.3f})")
                axes_pr[ri].grid(alpha=0.3)

            fig_roc.suptitle("Autoreg Presence Prediction — ROC per Round", fontsize=14)
            fig_roc.tight_layout(rect=[0,0,1,.94])
            fig_roc.savefig(os.path.join(out_dir, "presence_roc_per_round.png"),
                            dpi=150, bbox_inches="tight")
            plt.close(fig_roc)

            fig_pr.suptitle("Autoreg Presence Prediction — PR per Round", fontsize=14)
            fig_pr.tight_layout(rect=[0,0,1,.94])
            fig_pr.savefig(os.path.join(out_dir, "presence_pr_per_round.png"),
                           dpi=150, bbox_inches="tight")
            plt.close(fig_pr)
            log.info("Saved presence ROC and PR plots")

    # ========== UMAP ==========
    lat_path = os.path.join(args.output_dir, "inference", "latents.npz")
    if os.path.exists(lat_path):
        try:
            import umap
            lat = np.load(lat_path)
            z = lat.get("z", lat.get("mu", None))
            if z is not None and len(z) > 100:
                log.info("Computing UMAP...")
                max_pts = min(50000, len(z))
                idx = np.random.choice(len(z), max_pts, replace=False)
                emb = umap.UMAP(n_components=2, random_state=args.seed).fit_transform(z[idx])

                # Color by enrichment and z-norm
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                dec_e = lat["dec_enrich"]
                for ri, rname in enumerate(ROUND_NAMES):
                    sc = axes[ri].scatter(emb[:,0], emb[:,1],
                                          c=np.log1p(dec_e[idx, ri]),
                                          cmap="viridis", s=1, alpha=0.5)
                    axes[ri].set_title(f"UMAP — log(1+enrichment_{rname})")
                    plt.colorbar(sc, ax=axes[ri])
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, "umap_latent.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)
                np.savez_compressed(os.path.join(out_dir, "umap_coords.npz"), coords=emb, indices=idx)
                log.info("Saved UMAP plot")
        except ImportError:
            log.warning("umap-learn not installed — skipping UMAP.")

    log.info("Analysis complete -> %s", out_dir)


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args(); set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        log.info("Mixed precision enabled")
    if args.xla: log.info("XLA enabled")

    if _gpus: log.info("GPUs: %s", [g.name for g in _gpus])
    else: log.info("No GPU — CPU mode")

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if tfrecords_ready(args) and not args.force_tfrecord:
        log.info("TFRecords found — skipping preprocessing"); meta = load_meta(args)
    else:
        log.info("Preprocessing FASTA -> TFRecords..."); meta = preprocess_data(args)

    log.info("Data: n_total=%d n_train=%d n_val=%d n_test=%d",
             meta["n_total"], meta["n_train"], meta["n_val"], meta["n_test"])

    if args.mlm: train_mlm(args, meta)
    if args.autoregressive_vae: train_autoreg_vae(args, meta)
    if args.inference: run_inference(args, meta)
    if args.generator: run_generator(args, meta)
    if args.analysis: run_analysis(args, meta)

    if not any([args.mlm, args.autoregressive_vae, args.inference, args.generator, args.analysis]):
        log.info("No stage selected. Use --mlm, --autoregressive_vae, --inference, --generator, or --analysis.")


if __name__ == "__main__":
    main()