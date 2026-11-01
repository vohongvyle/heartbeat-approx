"""
Load the MIT-BIH Arrhythmia Database into the same format as synth_ecg.make_dataset.

Needs:  pip install wfdb   (downloads ~100 MB from PhysioNet on first use, cached in ./mitdb)

Standard choices (de Chazal et al., 2004), so results compare with the literature:
  * Lead MLII, AAMI classes N / S / V / F (Q and paced records are left out).
  * Inter-patient split: DS1 records for training (we hold out 5 of them for validation),
    DS2 records for testing. Never mix beats of one patient across train and test.
  * Signals downsampled 360 -> 180 Hz; window 0.25 s before to 0.46 s after the R peak.
  * Rhythm features: previous and next RR interval divided by the local mean RR
    (mean of the 10 surrounding RR intervals).

NOTE: written to match synth_ecg's format but not run in the environment where it was
written (no PhysioNet access there). Run `python mitbih.py` once to check the counts:
DS1 should give roughly 45.8k N, 0.9k S, 3.8k V, 0.4k F beats.
"""
import os
import numpy as np

AAMI = {**{s: 0 for s in "NLRej"}, **{s: 1 for s in "aAJS"}, **{s: 2 for s in "VE"}, "F": 3}
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208,
       209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222,
       228, 231, 232, 233, 234]
PRE, POST = 45, 83
FS_OUT = 180


def load_record(rec, folder="mitdb"):
    import wfdb
    if not os.path.exists(os.path.join(folder, f"{rec}.dat")):
        wfdb.dl_database("mitdb", folder, records=[str(rec)])
    sig, fields = wfdb.rdsamp(os.path.join(folder, str(rec)))
    lead = fields["sig_name"].index("MLII") if "MLII" in fields["sig_name"] else 0
    x = sig[:, lead]
    x = 0.5 * (x[0::2][: len(x) // 2] + x[1::2][: len(x) // 2])          # 360 -> 180 Hz, simple anti-alias
    ann = wfdb.rdann(os.path.join(folder, str(rec)), "atr")
    keep = [(s // 2, sym) for s, sym in zip(ann.sample, ann.symbol) if sym in AAMI or sym in "/fQ"]
    r = np.array([k[0] for k in keep]); sym = [k[1] for k in keep]
    rr = np.diff(r) / FS_OUT
    X, RR, y = [], [], []
    for i in range(1, len(r) - 1):
        if sym[i] not in AAMI or r[i] - PRE < 0 or r[i] + POST > len(x):
            continue
        lo, hi = max(0, i - 6), min(len(rr), i + 5)
        avg = np.mean(rr[lo:hi])
        beat = x[r[i] - PRE: r[i] + POST]
        X.append(beat - np.median(beat)); RR.append([rr[i - 1] / avg, rr[i] / avg]); y.append(AAMI[sym[i]])
    return np.array(X, np.float32), np.array(RR, np.float32), np.array(y)


def make_dataset(folder="mitdb", n_val_records=5, seed=0):
    """Same outputs as synth_ecg.make_dataset, plus fixed train / val / test masks."""
    recs = DS1 + DS2
    parts = [load_record(r, folder) for r in recs]
    X = np.concatenate([p[0] for p in parts]); RR = np.concatenate([p[1] for p in parts])
    y = np.concatenate([p[2] for p in parts])
    pid = np.concatenate([np.full(len(p[2]), r) for p, r in zip(parts, recs)])
    val_recs = list(np.random.default_rng(seed).choice(DS1, n_val_records, replace=False))
    masks = [np.isin(pid, [r for r in DS1 if r not in val_recs]), np.isin(pid, val_recs), np.isin(pid, DS2)]
    return X, RR, y, pid, masks


if __name__ == "__main__":
    X, RR, y, pid, (tr, va, te) = make_dataset()
    for name, m in [("train", tr), ("val", va), ("test (DS2)", te)]:
        print(name, {c: int((y[m] == i).sum()) for i, c in enumerate("NSVF")})
