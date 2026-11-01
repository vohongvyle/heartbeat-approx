"""
Export the trained ECG model, test heartbeats and multiplier tables for the web page.
Uses exactly the same data, training and calibration as ecg_experiment.py (same seed),
so the page reproduces the experiment's numbers.

    python3 ecg/export_web.py                 # synthetic data -> docs/data.js
    python3 ecg/export_web.py --data mitbih   # after ecg_experiment.py --data mitbih
"""
import argparse, base64, csv, gzip, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnn import CNN, QuantCNN, _sm  # noqa: E402
import ecg_experiment as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "mitbih"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--luts", default=os.path.join(HERE, "..", "luts"))
    ap.add_argument("--max-test", type=int, default=6000, help="cap on test beats shipped to the page")
    args = ap.parse_args()

    X, RR, y, (tr, va, te) = E.load(args)
    cnt = np.bincount(y[tr], minlength=4)
    cw = np.sqrt(cnt.sum() / np.maximum(cnt, 1)); cw /= cw[0]
    model = CNN(args.seed).fit(X[tr], RR[tr], y[tr], epochs=args.epochs, class_weight=cw, seed=args.seed)

    meta = json.load(open(os.path.join(args.luts, "circuits.json")))
    names = sorted(meta, key=lambda n: -meta[n]["pwr"])
    luts = np.stack([np.load(os.path.join(args.luts, n + ".npy")) for n in names]).astype(np.int64)
    ex = names.index(E.EXACT)
    q = QuantCNN(model, X[tr], RR[tr], luts)
    q.calibrate(X[tr][:3000], RR[tr][:3000], ex)

    idx = np.where(te)[0]
    if len(idx) > args.max_test:     # keep every abnormal beat, subsample normal ones
        rng = np.random.default_rng(0)
        ab, nn = idx[y[idx] != 0], idx[y[idx] == 0]
        idx = np.sort(np.concatenate([ab, rng.choice(nn, args.max_test - len(ab), replace=False)]))
    Xt, Rt, yt = X[idx], RR[idx], y[idx]

    # python reference predictions, for checking the page reproduces them
    cfg_exact = [ex] * 4
    ref_exact = q.logits(Xt, Rt, cfg_exact).argmax(1)
    search = json.load(open(os.path.join(HERE, "results", f"search_{args.data}.json")))
    presets = {k: [[names.index(c), bool(f)] for c, f in search[k]["config"]] for k in ("accuracy", "clinical")}
    acc_cfg = presets["accuracy"]
    ref_acc = q.logits(Xt, Rt, [c for c, _ in acc_cfg], [f for _, f in acc_cfg]).argmax(1)

    m1, s1 = _sm(Xt, q.L[0]["sa"])                                 # layer-1 input, sign-magnitude
    layers = [dict(name=L["name"], kind=L["kind"], K=L["K"], stride=L["stride"], sa=float(L["sa"]), sw=float(L["sw"]),
                   w=(L["mag"] * L["sgn"]).astype(int).flatten().tolist(), rows=int(L["mag"].shape[0]),
                   cols=int(L["mag"].shape[1]), b=L["b"].astype(float).tolist(),
                   corr=L["corr"].astype(float).tolist()) for L in q.L]
    sweep = list(csv.DictReader(open(os.path.join(HERE, "results", f"sweep_{args.data}.csv"))))
    d = dict(
        data=args.data, classes=["N", "S", "V", "F"], macs=q.macs, adder=E.ADDER_PWR,
        circuits=[dict(name=n, pwr=meta[n]["pwr"], mae=meta[n]["mae%"]) for n in names],
        layers=layers,
        x_b64=base64.b64encode((m1 * s1).astype("<i2").tobytes()).decode(), n_test=int(len(yt)),
        rr=np.round(Rt, 4).flatten().tolist(), y=yt.tolist(),
        presets=presets,
        sweep=[dict(circuit=r["circuit"], energy=float(r["energy"]), acc=float(r["acc"]), se_S=float(r["se_S"]),
                    se_V=float(r["se_V"])) for r in sweep],
        ref=dict(exact=ref_exact.tolist(), accuracy=ref_acc.tolist()),
    )
    blob = base64.b64encode(gzip.compress(luts.astype("<u2").tobytes(), 9)).decode()
    # the web page lives in docs/ (index.html, style.css, app.js) so GitHub Pages can serve it;
    # this script only rewrites docs/data.js
    docs = os.path.join(HERE, "..", "docs")
    os.makedirs(docs, exist_ok=True)
    path = os.path.join(docs, "data.js")
    with open(path, "w") as f:
        f.write("window.ECG=" + json.dumps(d, separators=(",", ":")) + ";\nwindow.LUTS_GZ='" + blob + "';\n")
    print(f"wrote {os.path.normpath(path)} ({os.path.getsize(path) / 1e6:.1f} MB, {len(yt)} test beats); "
          f"open docs/index.html in a browser")


if __name__ == "__main__":
    main()
