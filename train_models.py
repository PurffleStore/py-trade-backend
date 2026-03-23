"""Train AI TA ExtraTrees models for a list of symbols and save them under `models/`.

Usage:
  python train_models.py RELIANCE.NS TCS.NS
or
  python train_models.py --from-file symbols.txt

The script requires scikit-learn and joblib to be installed in your environment.
"""
import sys
import os
from pathlib import Path
from typing import List

from ai_ta_model import train_and_save_model


def train_symbols(symbols: List[str], out_dir: str = "models"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for sym in symbols:
        try:
            out_path = os.path.join(out_dir, f"{sym.replace('/', '_')}.pkl")
            print(f"Training model for {sym} -> {out_path} ...")
            meta = train_and_save_model(sym, out_path, model_type="extra_trees")
            print(f"  OK: accuracy={meta.get('accuracy')}, samples={meta.get('n_samples')}")
            results.append({"symbol": sym, "ok": True, "meta": meta})
        except Exception as e:
            print(f"  FAIL for {sym}: {type(e).__name__}: {e}")
            results.append({"symbol": sym, "ok": False, "error": f"{type(e).__name__}: {e}"})
    return results


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage: python train_models.py SYMBOL [SYMBOL ...]  OR  python train_models.py --from-file symbols.txt")
        sys.exit(1)

    args = sys.argv[1:]
    symbols = []
    if args[0] == "--from-file" and len(args) >= 2:
        fn = args[1]
        with open(fn, "r", encoding="utf-8") as f:
            symbols = [line.strip() for line in f if line.strip()]
    else:
        symbols = args

    res = train_symbols(symbols)
    print("Summary:")
    for r in res:
        if r["ok"]:
            print(f"  {r['symbol']}: OK")
        else:
            print(f"  {r['symbol']}: FAIL - {r['error']}")
