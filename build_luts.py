"""
Step 1: turn every 8x8 unsigned multiplier in EvoApproxLib into a 256x256 lookup table.

For each circuit we compile its C model into a small shared library, evaluate all
65,536 input pairs once, and save the table. Simulating a DNN with the approximate
multiplier is then just a table lookup instead of a real multiplication.

Usage:
    git clone https://github.com/ehw-fit/evoapproxlib
    python build_luts.py --lib ./evoapproxlib --out luts
"""
import argparse, ctypes, json, os, subprocess, tempfile
import numpy as np


def find_circuits(meta, kind="8x8_unsigned"):
    """Collect unique circuits (name -> params, C path) from meta.json."""
    found = {}

    def walk(o):
        if isinstance(o, dict):
            if "instances" in o and kind in o.get("folder", ""):
                for inst in o["instances"]:
                    cfile = next((f["file"] for f in inst["files"] if f["file"].endswith(".c")), None)
                    if cfile and inst["name"] not in found:
                        found[inst["name"]] = {
                            "params": {k: float(v) for k, v in inst["params"].items()},
                            "c_path": os.path.join(o["folder"], cfile),
                        }
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(meta)
    return found


def build_lut(c_path, func_name, workdir):
    so = os.path.join(workdir, func_name + ".so")
    subprocess.run(["gcc", "-O2", "-shared", "-fPIC", c_path, "-o", so], check=True)
    lib = ctypes.CDLL(so)
    f = getattr(lib, func_name)
    f.restype = ctypes.c_uint64
    f.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    lut = np.empty((256, 256), dtype=np.int32)
    for a in range(256):
        for b in range(256):
            lut[a, b] = f(a, b)
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="evoapproxlib")
    ap.add_argument("--out", default="luts")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.lib, "meta.json")))
    circuits = find_circuits(meta)
    os.makedirs(args.out, exist_ok=True)
    exact = np.outer(np.arange(256), np.arange(256)).astype(np.int64)

    info = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, c in sorted(circuits.items()):
            lut = build_lut(os.path.join(args.lib, c["c_path"]), name, tmp)
            np.save(os.path.join(args.out, name + ".npy"), lut)
            err = lut - exact
            info[name] = dict(c["params"],
                              measured_mae=float(np.abs(err).mean()),
                              measured_bias=float(err.mean()))
            print(f"{name:12s} pwr={c['params']['pwr']:.4f} mW  MAE%={c['params']['mae%']:<8} "
                  f"mean error={err.mean():+.1f}")

    json.dump(info, open(os.path.join(args.out, "circuits.json"), "w"), indent=1)
    print(f"\n{len(info)} lookup tables written to {args.out}/")


if __name__ == "__main__":
    main()
