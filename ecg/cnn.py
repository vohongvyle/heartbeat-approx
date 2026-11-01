"""
A small 1D CNN for heartbeat classification, in plain NumPy, plus its 8-bit
approximate-multiplier version.

Architecture (about 27k multiplications per beat, small enough for a wearable MCU):
    beat (128 samples)  -> conv 7 taps, 8 ch, stride 2 -> ReLU -> max-pool 2   (61 -> 30)
                        -> conv 5 taps, 16 ch          -> ReLU -> max-pool 2   (26 -> 13)
    flatten (208) + 2 rhythm features -> dense 32 -> ReLU -> dense 4 (N, S, V, F)

Activations are kept channels-last, (beats, positions, channels). A conv is an
"im2col" matrix product: each output position multiplies a patch of K x C inputs
with a (K*C, C_out) weight matrix. That makes conv and dense layers the same
operation, so one approximate matmul serves both.
"""
import numpy as np


# ----------------------------------------------------------------------------- helpers
def im2col(h, K, stride):
    """h: (N, L, C) -> patches (N, L_out, K*C)."""
    N, L, C = h.shape
    pos = np.arange(0, L - K + 1, stride)
    idx = pos[:, None] + np.arange(K)[None]
    return h[:, idx, :].reshape(N, len(pos), K * C), idx


def col2im(dP, idx, shape):
    N, L, C = shape
    dh = np.zeros(shape, dP.dtype)
    np.add.at(dh, (slice(None), idx), dP.reshape(N, idx.shape[0], idx.shape[1], C))
    return dh


def maxpool2(a):
    N, L, C = a.shape
    r = a[:, : L // 2 * 2].reshape(N, L // 2, 2, C)
    am = r.argmax(2)
    return np.take_along_axis(r, am[:, :, None], 2)[:, :, 0], am


def maxpool2_back(d, am, L):
    N, Lp, C = d.shape
    out = np.zeros((N, Lp, 2, C), d.dtype)
    np.put_along_axis(out, am[:, :, None], d[:, :, None], 2)
    full = np.zeros((N, L, C), d.dtype)
    full[:, : Lp * 2] = out.reshape(N, Lp * 2, C)
    return full


LAYERS = [  # name, kind, K, stride, C_in, C_out
    ("conv1", "conv", 7, 2, 1, 8),
    ("conv2", "conv", 5, 1, 8, 16),
    ("fc1", "fc", None, None, 13 * 16 + 2, 32),
    ("fc2", "fc", None, None, 32, 4),
]


# ----------------------------------------------------------------------------- float model
class CNN:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.p = {}
        for name, kind, K, s, cin, cout in LAYERS:
            fan_in = (K * cin) if kind == "conv" else cin
            self.p[name + ".W"] = rng.normal(0, np.sqrt(2 / fan_in), (fan_in, cout)).astype(np.float32)
            self.p[name + ".b"] = np.zeros(cout, np.float32)

    def forward(self, x, rr, train=False):
        p, c = self.p, {}
        P1, i1 = im2col(x[:, :, None], 7, 2); z1 = P1 @ p["conv1.W"] + p["conv1.b"]
        a1 = np.maximum(z1, 0); m1, am1 = maxpool2(a1)
        P2, i2 = im2col(m1, 5, 1); z2 = P2 @ p["conv2.W"] + p["conv2.b"]
        a2 = np.maximum(z2, 0); m2, am2 = maxpool2(a2)
        f = np.concatenate([m2.reshape(len(x), -1), rr], 1)
        z3 = f @ p["fc1.W"] + p["fc1.b"]; a3 = np.maximum(z3, 0)
        z4 = a3 @ p["fc2.W"] + p["fc2.b"]
        if train:
            c.update(x=x, P1=P1, i1=i1, z1=z1, am1=am1, m1=m1, P2=P2, i2=i2, z2=z2, am2=am2, f=f, z3=z3, a3=a3)
            return z4, c
        return z4

    def backward(self, dz4, c):
        p, g = self.p, {}
        g["fc2.W"] = c["a3"].T @ dz4; g["fc2.b"] = dz4.sum(0)
        dz3 = (dz4 @ p["fc2.W"].T) * (c["z3"] > 0)
        g["fc1.W"] = c["f"].T @ dz3; g["fc1.b"] = dz3.sum(0)
        df = dz3 @ p["fc1.W"].T
        dm2 = df[:, :208].reshape(-1, 13, 16)
        dz2 = maxpool2_back(dm2, c["am2"], c["z2"].shape[1]) * (c["z2"] > 0)
        g["conv2.W"] = c["P2"].reshape(-1, 40).T @ dz2.reshape(-1, 16); g["conv2.b"] = dz2.sum((0, 1))
        dm1 = col2im(dz2 @ p["conv2.W"].T, c["i2"], c["m1"].shape)
        dz1 = maxpool2_back(dm1, c["am1"], c["z1"].shape[1]) * (c["z1"] > 0)
        g["conv1.W"] = c["P1"].reshape(-1, 7).T @ dz1.reshape(-1, 8); g["conv1.b"] = dz1.sum((0, 1))
        return g

    def fit(self, X, RR, y, epochs=30, lr=2e-3, batch=128, seed=0, class_weight=None, verbose=False):
        rng = np.random.default_rng(seed)
        w = np.ones(4, np.float32) if class_weight is None else np.asarray(class_weight, np.float32)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}; v = {k: np.zeros_like(x) for k, x in self.p.items()}
        t = 0
        for ep in range(epochs):
            order = rng.permutation(len(X)); loss = 0
            for i in range(0, len(X), batch):
                b = order[i:i + batch]
                z, c = self.forward(X[b], RR[b], train=True)
                z = z - z.max(1, keepdims=True); pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
                wb = w[y[b]]; norm = wb.sum()
                loss += -(wb * np.log(pr[np.arange(len(b)), y[b]] + 1e-9)).sum() / norm * len(b)
                d = pr.copy(); d[np.arange(len(b)), y[b]] -= 1; d *= (wb / norm)[:, None]
                g = self.backward(d.astype(np.float32), c)
                t += 1
                for k in self.p:                                    # Adam
                    m[k] = 0.9 * m[k] + 0.1 * g[k]; v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                    self.p[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            if verbose:
                print(f"epoch {ep + 1:2d}  loss {loss / len(X):.4f}")
        return self


# ----------------------------------------------------------------------------- 8-bit approximate model
def _sm(a, s):
    """Sign-magnitude 8-bit: magnitude 0..255 and sign (+1 for zero, like hardware)."""
    return np.clip(np.round(np.abs(a) / s), 0, 255).astype(np.int64), np.where(a < 0, -1, 1)


class QuantCNN:
    """Every multiplication is a lookup in a 256x256 table of an EvoApproxLib 8x8 unsigned
    multiplier, applied to magnitudes; signs are handled separately (sign-magnitude), so
    the signed ECG input and signed weights work with an unsigned multiplier."""

    def __init__(self, cnn, X, RR, luts):
        self.luts = luts
        self.L = []
        for name, kind, K, s, cin, cout in LAYERS:
            W = cnn.p[name + ".W"]
            sw = np.abs(W).max() / 255
            mag, sgn = _sm(W, sw)
            self.L.append(dict(name=name, kind=kind, K=K, stride=s, sw=sw, mag=mag, sgn=sgn, b=cnn.p[name + ".b"]))
        self.macs = [3416, 16640, 210 * 32, 32 * 4]                 # multiplications per beat
        # activation scales: 99.9th percentile of |input| of each layer, exact float model
        ins = self._layer_inputs_float(cnn, X, RR)
        for L, a in zip(self.L, ins):
            L["sa"] = max(np.percentile(np.abs(a), 99.9), 1e-6) / 255

    @staticmethod
    def _layer_inputs_float(cnn, X, RR):
        p = cnn.p
        P1, _ = im2col(X[:, :, None], 7, 2)
        m1, _ = maxpool2(np.maximum(P1 @ p["conv1.W"] + p["conv1.b"], 0))
        P2, _ = im2col(m1, 5, 1)
        m2, _ = maxpool2(np.maximum(P2 @ p["conv2.W"] + p["conv2.b"], 0))
        f = np.concatenate([m2.reshape(len(X), -1), RR], 1)
        a3 = np.maximum(f @ p["fc1.W"] + p["fc1.b"], 0)
        return [P1, P2, f, a3]

    def _matmul(self, k, A, c, comp, rows=4096):
        """A: (M, R) float inputs of layer k -> (M, C_out) float pre-activations."""
        L = self.L[k]; lut = self.luts[c]
        mag, sgn = _sm(A, L["sa"])
        out = np.empty((len(A), L["mag"].shape[1]), np.int64)
        for i in range(0, len(A), rows):
            pm = lut[mag[i:i + rows, :, None], L["mag"][None]]      # (m, R, C_out) table lookups
            out[i:i + rows] = (pm * (sgn[i:i + rows, :, None] * L["sgn"][None])).sum(1)
        if comp:
            out = out - L["corr"][c]
        return out * (L["sa"] * L["sw"]) + L["b"]

    def logits(self, X, RR, circuits, comp=(False,) * 4):
        N = len(X)
        P1, _ = im2col(X[:, :, None], 7, 2)
        z1 = self._matmul(0, P1.reshape(-1, 7), circuits[0], comp[0]).reshape(N, -1, 8)
        m1, _ = maxpool2(np.maximum(z1, 0))
        P2, _ = im2col(m1, 5, 1)
        z2 = self._matmul(1, P2.reshape(-1, 40), circuits[1], comp[1]).reshape(N, -1, 16)
        m2, _ = maxpool2(np.maximum(z2, 0))
        f = np.concatenate([m2.reshape(N, -1), RR], 1)
        a3 = np.maximum(self._matmul(2, f, circuits[2], comp[2]), 0)
        return self._matmul(3, a3, circuits[3], comp[3])

    def calibrate(self, X, RR, exact_idx):
        """Bias compensation, folded into each neuron's bias (no extra hardware).
        For every input column r of a layer we measure the distribution of its signed
        8-bit value on calibration data (exact upstream layers). The expected error of a
        product with weight magnitude m is then E_r(m) = sum_a h_r(a) * err(a, m), where
        h_r(a) = P(|x|=a, x>=0) - P(|x|=a, x<0) carries the activation sign. Each neuron
        gets corr = sum_r sign(w_r) * E_r(|w_r|) for every candidate circuit."""
        exact = np.outer(np.arange(256), np.arange(256))
        err = self.luts - exact                                     # (C, 256, 256)
        N = len(X)
        inputs = []
        P1, _ = im2col(X[:, :, None], 7, 2); inputs.append(P1.reshape(-1, 7))
        z1 = self._matmul(0, inputs[0], exact_idx, False).reshape(N, -1, 8)
        m1, _ = maxpool2(np.maximum(z1, 0))
        P2, _ = im2col(m1, 5, 1); inputs.append(P2.reshape(-1, 40))
        z2 = self._matmul(1, inputs[1], exact_idx, False).reshape(N, -1, 16)
        m2, _ = maxpool2(np.maximum(z2, 0))
        inputs.append(np.concatenate([m2.reshape(N, -1), RR], 1))
        inputs.append(np.maximum(self._matmul(2, inputs[2], exact_idx, False), 0))
        for L, A in zip(self.L, inputs):
            mag, sgn = _sm(A, L["sa"])
            h = np.zeros((A.shape[1], 256))
            for r in range(A.shape[1]):
                h[r] = np.bincount(mag[:, r], weights=sgn[:, r], minlength=256) / len(A)
            Eim = np.einsum("ra,cam->crm", h, err)                  # (C, R, 256)
            rows = np.arange(L["mag"].shape[0])[:, None]
            L["corr"] = (L["sgn"][None] * Eim[:, rows, L["mag"]]).sum(1)   # (C, C_out)
