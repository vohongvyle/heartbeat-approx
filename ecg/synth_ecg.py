"""
Synthetic heartbeats for testing the pipeline without downloading MIT-BIH.

Each beat is a 128-sample window (180 Hz, 0.25 s before to 0.46 s after the R peak)
built from Gaussian waves (P, Q, R, S, T), plus two rhythm features. Classes follow
the AAMI convention used with MIT-BIH:

    N  normal            regular timing, normal P-QRS-T
    S  supraventricular  premature, abnormal or missing P wave, normal narrow QRS
    V  ventricular       premature, no P wave, wide QRS, T wave of opposite polarity,
                         compensatory pause after it
    F  fusion            a blend of normal and ventricular shape, mildly premature

Realism that matters for the experiments:
  * Beats come from simulated patients, each with their own heart rate, wave shapes,
    noise and ventricular-beat shape. Splits are by patient (inter-patient), as in the
    standard MIT-BIH DS1/DS2 evaluation, so test patients look new to the model.
  * Abnormal beats are rare (about 85% N) and S beats are often only slightly premature,
    so S is the hardest class, as in the real data.
  * When the previous beat is close, its T wave shows up at the left edge of the window.

This is a test bench, not a substitute for real ECG. Do not report its numbers as results.
"""
import numpy as np

FS = 180                      # Hz after downsampling
PRE, POST = 45, 83            # samples before / after R  -> 128-sample window
CLASSES = ["N", "S", "V", "F"]
T_AXIS = (np.arange(PRE + POST) - PRE) / FS


def _g(t, c, a, w):
    return a * np.exp(-0.5 * ((t - c) / w) ** 2)


def _normal_shape(t, p):
    return (_g(t, -0.17, p["P"], 0.022) + _g(t, -0.025, -0.12 * p["amp"], 0.008)
            + _g(t, 0.0, p["amp"], 0.010 * p["qrs_w"]) + _g(t, 0.025, -0.22 * p["amp"], 0.009 * p["qrs_w"])
            + _g(t, 0.24, p["T"], 0.04))


def _ventricular_shape(t, p, rng):
    w = p["pvc_w"] * rng.uniform(0.9, 1.1)
    a = p["pvc_amp"] * rng.uniform(0.85, 1.15)
    return (_g(t, 0.0, a, 0.010 * w) + _g(t, 0.05 * w / 3, -0.5 * a, 0.012 * w)
            + _g(t, 0.29, -np.sign(a) * p["pvc_T"], 0.06))


def _patient(rng):
    return dict(
        rr=60 / rng.uniform(58, 100),                       # mean RR, seconds
        amp=rng.uniform(0.6, 1.4), qrs_w=rng.uniform(0.85, 1.2),
        P=rng.uniform(0.06, 0.18), T=rng.uniform(0.12, 0.45) * rng.choice([1, 1, 1, -1]),
        pvc_w=rng.uniform(2.3, 4.0), pvc_amp=rng.uniform(0.8, 1.7) * rng.choice([1, -1]),
        pvc_T=rng.uniform(0.25, 0.6),
        noise=rng.uniform(0.015, 0.05), wander=rng.uniform(0.02, 0.15),
        # class mix differs a lot between patients, as in MIT-BIH
        mix=rng.dirichlet([18, 1.5, 2.5, 0.5]) if rng.random() < 0.7 else rng.dirichlet([40, 0.1, 0.1, 0.05]),
    )


def _beat(cls, p, rng):
    t = T_AXIS
    q = dict(p, amp=p["amp"] * rng.uniform(0.93, 1.07), P=p["P"] * rng.uniform(0.8, 1.2),
             T=p["T"] * rng.uniform(0.85, 1.15))
    if cls == "N":
        pre = p["rr"] * rng.normal(1, 0.06); post = p["rr"] * rng.normal(1, 0.06)
        x = _normal_shape(t, q)
    elif cls == "S":
        pre = p["rr"] * (rng.uniform(0.84, 0.95) if rng.random() < 0.35 else rng.uniform(0.55, 0.84))
        post = p["rr"] * rng.uniform(0.98, 1.2)
        q["P"] = q["P"] * rng.uniform(-0.7, 0.7)                         # abnormal or flat P
        x = _normal_shape(t + rng.uniform(-0.01, 0.01), q)
    elif cls == "V":
        pre = p["rr"] * rng.uniform(0.5, 0.85); post = p["rr"] * rng.uniform(1.15, 1.55)
        x = _ventricular_shape(t, p, rng)
    else:  # F
        a = rng.uniform(0.3, 0.7)
        pre = p["rr"] * rng.uniform(0.78, 0.98); post = p["rr"] * rng.uniform(0.98, 1.2)
        x = a * _normal_shape(t, q) + (1 - a) * _ventricular_shape(t, p, rng)
    x = x + _g(t, -pre + 0.24, q["T"], 0.04)                             # previous beat's T wave
    x = x + p["wander"] * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t + rng.uniform(0, 6.3))
    x = x + rng.normal(0, p["noise"], t.shape)
    return x, pre, post


def make_dataset(n_patients=48, beats_per_patient=260, seed=0):
    """Returns X (n, 128) in mV, rr (n, 2) = [pre/avg, post/avg], y (n,) class index, patient (n,)."""
    rng = np.random.default_rng(seed)
    X, RR, y, pid = [], [], [], []
    for k in range(n_patients):
        p = _patient(rng)
        for _ in range(beats_per_patient):
            c = rng.choice(4, p=p["mix"])
            x, pre, post = _beat(CLASSES[c], p, rng)
            X.append(x); RR.append([pre / p["rr"], post / p["rr"]]); y.append(c); pid.append(k)
    return np.array(X, np.float32), np.array(RR, np.float32), np.array(y), np.array(pid)


def patient_split(pid, seed=0, frac=(0.5, 0.2)):
    """Split by patient: train / validation / test patient sets, no patient in two sets."""
    pats = np.random.default_rng(seed).permutation(np.unique(pid))
    a = int(len(pats) * frac[0]); b = a + int(len(pats) * frac[1])
    return [np.isin(pid, s) for s in (pats[:a], pats[a:b], pats[b:])]


if __name__ == "__main__":
    X, RR, y, pid = make_dataset()
    print("beats", X.shape, "class counts", {c: int((y == i).sum()) for i, c in enumerate(CLASSES)})
