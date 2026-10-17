#!/usr/bin/env python3
"""
Packs the curated base pair mmCIF files (data/curated/<class>/*.cif) into one binary file (data/curated.npz),
holding just what the RMSD search needs: file names, purine/pyrimidine flags and the specific-atom coordinates.
CuratedRMSDMetric reads this file, one class at a time, instead of parsing mmCIF files.

Rebuild after changing anything under data/curated:
    python curated_pack.py [curated_dir] [out.npz]
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_ATOMS = 4  # purines: C1', N9, C4, C8; pyrimidines: C1', N1, C2, C6


def pack_path_for(curated_dir: str) -> str:
    return os.path.normpath(curated_dir) + ".npz"


def build(curated_dir: str, out_path: str) -> int:
    import pairs
    metric = pairs.CuratedRMSDMetric(use_knn_scoring=False, curated_dir=curated_dir)
    arrays = {}
    total = 0
    for cls in sorted(os.listdir(curated_dir)):
        class_dir = os.path.join(curated_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        names, purine, counts, xyz = [], [], [], []
        for fname in sorted(f for f in os.listdir(class_dir) if f.endswith(".cif")):
            ref = metric.get_reference(os.path.join(class_dir, fname))
            if ref is None:
                continue
            t1, t2, c1, c2 = ref
            block = np.full((2, MAX_ATOMS, 3), np.nan)
            block[0, :len(c1)] = c1
            block[1, :len(c2)] = c2
            names.append(fname[:-4])
            purine.append((t1 == "purine", t2 == "purine"))
            counts.append((len(c1), len(c2)))
            xyz.append(block)
        if names:
            arrays[f"{cls}/names"] = np.array(names)
            arrays[f"{cls}/purine"] = np.array(purine, dtype=bool)
            arrays[f"{cls}/counts"] = np.array(counts, dtype=np.int8)
            arrays[f"{cls}/xyz"] = np.array(xyz)
            total += len(names)
    np.savez_compressed(out_path, **arrays)
    return total


if __name__ == "__main__":
    import pairs
    curated = sys.argv[1] if len(sys.argv) > 1 else pairs.DEFAULT_CURATED_DIR
    out = sys.argv[2] if len(sys.argv) > 2 else pack_path_for(curated)
    n = build(curated, out)
    print(f"Packed {n} curated pairs into {out} ({os.path.getsize(out) / 1e6:.2f} MB)")
