"""
Risk-aware approximate computing for wearable arrhythmia detection.

Experiments (all with EvoApproxLib 8x8 multipliers, energy = multiplier + exact 16-bit adder per MAC):
  1. Uniform sweep: the same circuit in every layer, with/without bias compensation.
     Reports accuracy AND per-class sensitivity (Se) for S, V and F beats.
  2. Layer-wise search under two different constraints, chosen on validation patients:
       "accuracy" : overall accuracy within --acc-budget points of exact
       "clinical" : the same, AND sensitivity for S and V beats within --se-budget points
     then both are scored on unseen test patients.
  3. Confidence-gated exact fallback: run the cheap model; re-run a beat on the exact
     multiplier when the cheap model is unsure or calls it abnormal.
  4. What the saving could mean for battery life, under a stated assumption.

Usage:
    python ecg_experiment.py                      # synthetic heartbeats (no download)
    python ecg_experiment.py --data mitbih        # MIT-BIH, needs `pip install wfdb`
"""
import argparse, csv, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnn import CNN, QuantCNN  # noqa: E402

CLASSES = ["N", "S", "V", "F"]
EXACT = "mul8u_1JFF"
ADDER_PWR = 0.072        # mW, exact 16-bit adder add16u_1E2


# ----------------------------------------------------------------------------- data
def load(args):
    if args.data == "synthetic":
        from synth_ecg import make_dataset, patient_split
        X, RR, y, pid = make_dataset(seed=args.seed)
        masks = patient_split(pid, seed=args.seed)
    else:
        from mitbih import make_dataset
        X, RR, y, pid, masks = make_dataset(seed=args.seed)
    X = X - np.median(X, 1, keepdims=True)
    tr = masks[0]
    X = X / X[tr].std()
    RR = (RR - RR[tr].mean(0)) / RR[tr].std(0)
    return X.astype(np.float32), RR.astype(np.float32), y, masks


def metrics(pred, y):
    m = {"acc": float((pred == y).mean())}
    for c, n in enumerate(CLASSES):
        sel = y == c
        m[f"se_{n}"] = float((pred[sel] == c).mean()) if sel.any() else float("nan")
        called = pred == c
        m[f"ppv_{n}"] = float((y[called] == c).mean()) if called.any() else float("nan")
    m["missed_abnormal"] = int(((y != 0) & (pred == 0)).sum())
    return m


# ----------------------------------------------------------------------------- experiment
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "mitbih"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--acc-budget", type=float, default=0.5, help="points of overall accuracy")
    ap.add_argument("--se-budget", type=float, default=1.0, help="points of S and V sensitivity")
    ap.add_argument("--luts", default=os.path.join(HERE, "..", "luts"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    X, RR, y, (tr, va, te) = load(args)
    print(f"{args.data}: train {tr.sum()} / val {va.sum()} / test {te.sum()} beats; "
          f"test classes {dict(zip(CLASSES, np.bincount(y[te], minlength=4).tolist()))}")
    cnt = np.bincount(y[tr], minlength=4)
    cw = np.sqrt(cnt.sum() / np.maximum(cnt, 1)); cw /= cw[0]          # softened class weights
    model = CNN(args.seed).fit(X[tr], RR[tr], y[tr], epochs=args.epochs, class_weight=cw, seed=args.seed)

    meta = json.load(open(os.path.join(args.luts, "circuits.json")))
    names = sorted(meta, key=lambda n: -meta[n]["pwr"])
    luts = np.stack([np.load(os.path.join(args.luts, n + ".npy")) for n in names]).astype(np.int64)
    pwr = np.array([meta[n]["pwr"] for n in names])
    ex = names.index(EXACT)
    q = QuantCNN(model, X[tr], RR[tr], luts)
    q.calibrate(X[tr][:3000], RR[tr][:3000], ex)
    macs = np.array(q.macs)
    e_exact = (macs * (pwr[ex] + ADDER_PWR)).sum()
    energy = lambda cfg: float((macs * (pwr[[c for c, _ in cfg]] + ADDER_PWR)).sum() / e_exact)

    Xv, Rv, yv, Xt, Rt, yt = X[va], RR[va], y[va], X[te], RR[te], y[te]
    run = lambda Xs, Rs, cfg: q.logits(Xs, Rs, [c for c, _ in cfg], [k for _, k in cfg])
    exact_cfg = [(ex, False)] * 4
    base_t = metrics(run(Xt, Rt, exact_cfg).argmax(1), yt)
    base_v = metrics(run(Xv, Rv, exact_cfg).argmax(1), yv)
    float_t = metrics(model.forward(Xt, Rt).argmax(1), yt)
    print(f"float test acc {float_t['acc']:.4f} | 8-bit exact test acc {base_t['acc']:.4f}, "
          f"Se S {base_t['se_S']:.3f}, Se V {base_t['se_V']:.3f}")

    # 1) uniform sweep ------------------------------------------------------------
    sweep = []
    for i, n in enumerate(names):
        row = dict(circuit=n, pwr_mW=meta[n]["pwr"], energy=round(energy([(i, False)] * 4), 4))
        for comp in (False, True):
            m = metrics(run(Xt, Rt, [(i, comp)] * 4).argmax(1), yt)
            tag = "comp_" if comp else ""
            row.update({tag + k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
        sweep.append(row)
        print(f"  {n:11s} E={row['energy']:.3f}  acc {row['acc']:.4f}  Se S {row['se_S']:.3f}  "
              f"Se V {row['se_V']:.3f}   | comp: acc {row['comp_acc']:.4f}  Se S {row['comp_se_S']:.3f}")
    write_csv(os.path.join(args.out, f"sweep_{args.data}.csv"), sweep)

    # 2) layer-wise greedy search under two constraints --------------------------------
    def ok_accuracy(m):
        return m["acc"] >= base_v["acc"] - args.acc_budget / 100

    def ok_clinical(m):
        return ok_accuracy(m) and all(m[f"se_{c}"] >= base_v[f"se_{c}"] - args.se_budget / 100 for c in ("S", "V"))

    options = sorted([(i, c) for i in range(len(names)) for c in (False, True)], key=lambda o: pwr[o[0]])
    searches = {}
    for label, ok in [("accuracy", ok_accuracy), ("clinical", ok_clinical)]:
        cfg = list(exact_cfg)
        for _ in range(2):                                            # two greedy passes
            for k in np.argsort(-macs):                               # biggest layer first
                for o in options:                                     # cheapest circuit first
                    if pwr[o[0]] >= pwr[cfg[k][0]]:
                        break
                    trial = cfg[:k] + [o] + cfg[k + 1:]
                    if ok(metrics(run(Xv, Rv, trial).argmax(1), yv)):
                        cfg = trial
                        break
        mt = metrics(run(Xt, Rt, cfg).argmax(1), yt)
        searches[label] = dict(config=[(names[c], bool(k)) for c, k in cfg], energy=energy(cfg),
                               val=metrics(run(Xv, Rv, cfg).argmax(1), yv), test=mt, cfg=cfg)
        print(f"  search[{label}]: energy {energy(cfg):.3f}  test acc {mt['acc']:.4f}  Se S {mt['se_S']:.3f}  "
              f"Se V {mt['se_V']:.3f}  missed abnormal {mt['missed_abnormal']}")
    with open(os.path.join(args.out, f"search_{args.data}.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "cfg"} for k, v in searches.items()}
                  | {"exact": {"val": base_v, "test": base_t}}, f, indent=1)

    # 3) confidence-gated exact fallback ---------------------------------------------
    cheap = searches["accuracy"]["cfg"]
    zc = run(Xt, Rt, cheap); pc = softmax(zc)
    ze = run(Xt, Rt, exact_cfg); pe = ze.argmax(1)
    e_cheap = energy(cheap)
    gate = []
    for tau in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.01]:
        # Keep every abnormal flag from the cheap model (letting the exact model overrule an
        # alarm can only add misses); re-check only "normal" calls made with low confidence.
        redo = (pc.argmax(1) == 0) & (pc.max(1) < tau)
        pred = np.where(redo, pe, pc.argmax(1))
        m = metrics(pred, yt)
        gate.append(dict(tau=tau, rerun_frac=round(float(redo.mean()), 4),
                         energy=round(e_cheap + float(redo.mean()), 4), acc=round(m["acc"], 4),
                         se_S=round(m["se_S"], 4), se_V=round(m["se_V"], 4),
                         missed_abnormal=m["missed_abnormal"]))
    write_csv(os.path.join(args.out, f"gated_{args.data}.csv"), gate)
    print("  gated fallback (cheap = accuracy-search config; re-check low-confidence normal calls):")
    for g in gate:
        print(f"    tau {g['tau']:<5} rerun {g['rerun_frac']:.3f}  energy {g['energy']:.3f}  "
              f"Se S {g['se_S']:.3f}  missed {g['missed_abnormal']}")

    # 4) battery illustration -------------------------------------------------------
    print("\n  Battery life gain if inference is a share f of the device's power (assumption):")
    for label, e in [("accuracy search", searches["accuracy"]["energy"]),
                     ("clinical search", searches["clinical"]["energy"])]:
        print(f"    {label:16s} " + "  ".join(f"f={f:.0%}: +{(1 / (1 - f * (1 - e)) - 1):.0%}" for f in (0.1, 0.3, 0.5)))

    plot(sweep, base_t, searches, gate, args)
    print(f"\nDone in {time.time() - t0:.0f} s. Results in {args.out}/")


def softmax(z):
    z = z - z.max(1, keepdims=True); p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def plot(sweep, base, searches, gate, args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9a"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=140)
    e = [r["energy"] for r in sweep]

    ax = axes[0]
    ax.scatter(e, [100 * r["acc"] for r in sweep], s=26, color=GREY, label="Overall accuracy", zorder=3)
    ax.scatter(e, [100 * r["se_S"] for r in sweep], s=26, color=ORANGE, label="Sensitivity, S beats", zorder=3)
    ax.scatter(e, [100 * r["se_V"] for r in sweep], s=26, color=BLUE, marker="s", label="Sensitivity, V beats", zorder=3)
    ax.set_ylim(40, 101); ax.set_xlim(0, 1.03)
    ax.set_title("Same circuit in every layer", loc="left", fontsize=10)
    ax.set_xlabel("MAC energy per beat, relative to exact"); ax.set_ylabel("%")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    labels = ["Exact", "Accuracy\nconstraint", "Clinical\nconstraint"]
    vals = [base] + [searches[k]["test"] for k in ("accuracy", "clinical")]
    xs = np.arange(3); w = 0.26
    for j, (key, col, nm) in enumerate([("acc", GREY, "Accuracy"), ("se_S", ORANGE, "Se S"), ("se_V", BLUE, "Se V")]):
        ax.bar(xs + (j - 1) * w, [100 * v[key] for v in vals], w * 0.92, color=col, label=nm)
    ens = [1.0] + [searches[k]["energy"] for k in ("accuracy", "clinical")]
    lo = max(0, min(100 * min(v[k] for v in vals for k in ("acc", "se_S", "se_V")) - 5, 90))
    ax.set_xticks(xs, [f"{l}\nenergy {e:.0%}" for l, e in zip(labels, ens)]); ax.set_ylim(lo, 104)
    ax.set_title("Layer-wise search, scored on unseen patients", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)

    ax = axes[2]
    ax.plot([g["energy"] for g in gate], [g["missed_abnormal"] for g in gate], "-o", color=GREEN, ms=5)
    for g in gate:
        if g["tau"] in (0.0, 0.9, 0.99, 1.01):
            ax.annotate("cheap only" if g["tau"] == 0 else ("re-check every normal call" if g["tau"] > 1 else f"τ={g['tau']}"),
                        (g["energy"], g["missed_abnormal"]), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.axhline(base["missed_abnormal"], color=GREY, ls="--", lw=1)
    ax.text(max(g["energy"] for g in gate), base["missed_abnormal"] + 0.3,
            "exact model alone", ha="right", va="bottom", fontsize=8, color="#555")
    ax.set_xlabel("MAC energy per beat (cheap model + re-runs), relative to exact")
    ax.set_ylabel("Abnormal beats called normal (test)")
    ax.set_title("Confidence-gated exact fallback", loc="left", fontsize=10)
    for a in axes:
        a.grid(True, color="#e8e8e8"); [a.spines[s].set_visible(False) for s in ("top", "right")]
    fig.suptitle(f"Risk-aware approximate computing for arrhythmia detection ({args.data} data)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"ecg_{args.data}.png"))


if __name__ == "__main__":
    main()
