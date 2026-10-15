#!/usr/bin/env python3
"""
Compares base pairs annotated in two structures (typically a reference and a model)
and reports PPV, sensitivity, MCC and F1.

MCC is computed as sqrt(PPV * STY), the interaction network fidelity (INF) of
Parisien et al. (2009), RNA 15:1875, as used in RNA-Puzzles.
F1 is the harmonic mean of PPV and STY: 2*TP / (2*TP + FP + FN).
"""
import argparse, json, math, os, re, sys, typing as ty
import polars as pl
import assign_in_structure
import pdb_utils

# (chain, number, insertion code)
ResidueKey = tuple[str, int, str]
# (residue1, residue2, family) with residue1 <= residue2
Interaction = tuple[ResidueKey, ResidueKey, str]

CANONICAL_BASES = {"GC", "CG", "AU", "UA", "AT", "TA", "GU", "UG", "GT", "TG"}

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def normalize_family(family: str) -> str | None:
    """'cWH' / 'tHs' / 'cWWa' -> 'cWH' / 'tHS' / 'cWW'; near pairs ('ncWW') -> None"""
    m = re.match(r"(?P<n>n)?(?P<cistrans>[ct])(?P<e1>[WHSB])(?P<e2>[WHSB])", family, re.IGNORECASE)
    if m is None:
        raise ValueError(f"Invalid family: {family}")
    if m.group("n"):
        return None
    return m.group("cistrans").lower() + m.group("e1").upper() + m.group("e2").upper()

def base_letter(resname: str) -> str:
    return assign_in_structure.pair_defs.map_resname(resname.upper())[-1:].upper()

# {chain: {(number, insertion code): base}}
ChainResidues = dict[str, dict[tuple[int, str], str]]

def chain_residues(df: pl.DataFrame, suffixes: list[str]) -> ChainResidues:
    result: ChainResidues = {}
    if len(df) == 0:
        return result
    df = df.filter(pl.col("model") == df["model"].min())
    for s in suffixes:
        for ch, nr, ins, res in df.select(f"auth_asym_id{s}", f"auth_seq_id{s}", f"pdbx_PDB_ins_code{s}", f"auth_comp_id{s}").iter_rows():
            result.setdefault(str(ch), {})[(int(nr), (ins or "").strip())] = base_letter(res)
    return result

def load_annotations(input: str, boundaries: pl.DataFrame, best_fit: str) -> tuple[pl.DataFrame, ChainResidues]:
    """
    Loads basepairs from an assign_in_structure output file (.csv/.parquet), or annotates a structure.
    Also returns the residues of each chain (only the paired ones for .csv/.parquet input).
    """
    if input.endswith(".csv") or input.endswith(".parquet"):
        if input.endswith(".csv"):
            df = pl.read_csv(input, infer_schema_length=10000, dtypes={"pdbx_PDB_ins_code1": pl.Utf8, "pdbx_PDB_ins_code2": pl.Utf8})
        else:
            df = pl.read_parquet(input)
        return df, chain_residues(df, ["1", "2"])
    result = assign_in_structure.analyze_structure(input, 4.0, boundaries, best_fit, False, None) # type: ignore
    if result is None:
        raise ValueError(f"Could not analyze {input}")
    _, df, all_residues = result
    return df, chain_residues(all_residues, [""])

def match_chains(ref: ChainResidues, model: ChainResidues) -> dict[str, str]:
    """
    Maps model chains to reference chains by the number of residues with the same number and base.
    Ties (e.g. homodimers) prefer chains of the same name.
    """
    candidates = sorted(
        (
            -sum(1 for k, b in mres.items() if rres.get(k) == b),
            mch != rch,
            mch, rch
        )
        for mch, mres in model.items()
        for rch, rres in ref.items()
    )
    mapping: dict[str, str] = {}
    for neg_score, _, mch, rch in candidates:
        if neg_score < 0 and mch not in mapping and rch not in mapping.values():
            mapping[mch] = rch
    return mapping

def parse_chain_map(s: str) -> dict[str, str]:
    """'C:A,D:B' -> {'C': 'A', 'D': 'B'}"""
    return dict(tuple(x.split(":", 1)) for x in s.split(",") if x) # type: ignore

def to_interactions(df: pl.DataFrame, ignore_family: bool, chain_map: dict[str, str] = {}) -> dict[Interaction, str]:
    """Returns {interaction: bases} for the first model, with chains renamed by `chain_map` and residues in a canonical order"""
    result: dict[Interaction, str] = {}
    if len(df) == 0:
        return result
    first_model = df["model"].min()
    for row in df.filter(pl.col("model") == first_model).iter_rows(named=True):
        family = normalize_family(row["family"])
        if family is None:
            continue
        r1 = (chain_map.get(str(row["auth_asym_id1"]), str(row["auth_asym_id1"])), int(row["auth_seq_id1"]), (row["pdbx_PDB_ins_code1"] or "").strip())
        r2 = (chain_map.get(str(row["auth_asym_id2"]), str(row["auth_asym_id2"])), int(row["auth_seq_id2"]), (row["pdbx_PDB_ins_code2"] or "").strip())
        bases = base_letter(row["auth_comp_id1"]) + base_letter(row["auth_comp_id2"])
        if r2 < r1:
            r1, r2 = r2, r1
            family = family[0] + family[2] + family[1]
            bases = bases[::-1]
        if ignore_family:
            family = "*"
        result[(r1, r2, family)] = bases
    return result

def is_canonical(interaction: Interaction, bases: str) -> bool:
    return interaction[2] in ("cWW", "*") and bases in CANONICAL_BASES

def scores(ref: set[Interaction], model: set[Interaction]) -> dict[str, ty.Any]:
    tp, fp, fn = len(ref & model), len(model - ref), len(ref - model)
    ppv = tp / (tp + fp) if tp + fp else None
    sty = tp / (tp + fn) if tp + fn else None
    mcc = math.sqrt(ppv * sty) if ppv is not None and sty is not None else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {"TP": tp, "FP": fp, "FN": fn, "PPV": ppv, "STY": sty, "MCC": mcc, "F1": f1}

def format_interaction(i: Interaction, bases: str) -> str:
    def res(r: ResidueKey): return f"{r[0]}{r[1]}{r[2]}"
    return f"{res(i[0])}-{res(i[1])} {i[2]} {bases[0]}-{bases[1]}"

def main(args):
    pdb_utils.set_pdb_cache_dirs(args.pdbcache)
    boundaries = pl.read_csv(args.boundaries, infer_schema_length=10000, dtypes={"subclass": pl.Utf8})

    ref_df, ref_residues = load_annotations(args.reference, boundaries, args.best_fit)
    model_df, model_residues = load_annotations(args.model, boundaries, args.best_fit)

    if args.chain_map is not None:
        chain_map = parse_chain_map(args.chain_map)
    elif args.no_chain_matching:
        chain_map = {}
    else:
        chain_map = match_chains(ref_residues, model_residues)
    renamed = {m: r for m, r in chain_map.items() if m != r}

    ref = to_interactions(ref_df, args.ignore_family)
    model = to_interactions(model_df, args.ignore_family, chain_map)
    bases = {**model, **ref}

    categories = {
        "all": lambda i: True,
        "canonical": lambda i: is_canonical(i, bases[i]),
        "non-canonical": lambda i: not is_canonical(i, bases[i]),
    }
    results = {
        name: scores({i for i in ref if f(i)}, {i for i in model if f(i)})
        for name, f in categories.items()
    }

    if args.json:
        json.dump({"reference": args.reference, "model": args.model, "chain_map": renamed, **results}, sys.stdout, indent=2)
        print()
    else:
        fmt = lambda x: "-" if x is None else f"{x:.3f}"
        print(f"reference: {args.reference}")
        print(f"model:     {args.model}")
        if renamed:
            print(f"chains:    {', '.join(f'{m}->{r}' for m, r in sorted(renamed.items()))} (model->reference)")
        print(f"{'category':<14} {'TP':>4} {'FP':>4} {'FN':>4} {'PPV':>6} {'STY':>6} {'MCC':>6} {'F1':>6}")
        for name, r in results.items():
            print(f"{name:<14} {r['TP']:>4} {r['FP']:>4} {r['FN']:>4} {fmt(r['PPV']):>6} {fmt(r['STY']):>6} {fmt(r['MCC']):>6} {fmt(r['F1']):>6}")

    if args.list:
        out = sys.stderr if args.json else sys.stdout
        for label, s in [("TP", ref.keys() & model.keys()), ("FP", model.keys() - ref.keys()), ("FN", ref.keys() - model.keys())]:
            for i in sorted(s):
                print(f"{label}\t{format_interaction(i, bases[i])}", file=out)

def parser():
    p = argparse.ArgumentParser(description="""
        Compares basepairs in two structures and computes MCC (INF = sqrt(PPV * STY)) and F1.
        Inputs may be mmCIF/PDB files, PDB IDs, or .csv/.parquet outputs of assign_in_structure.py.
        Residues are matched by auth chain ID, residue number and insertion code; only the first model is used.
        Model chains are renamed to the reference chains with the most residues of the same number and base.
        Near pairs (n*) are ignored.
        """)
    p.add_argument("reference", help="Reference (native) structure")
    p.add_argument("model", help="Model (predicted) structure")
    p.add_argument("--ignore-family", action="store_true", help="Count a pair as matched regardless of its Leontis-Westhof family")
    p.add_argument("--chain-map", help="Explicit mapping of model chains to reference chains, e.g. 'C:A,D:B'")
    p.add_argument("--no-chain-matching", action="store_true", help="Match chains by name only")
    p.add_argument("--list", action="store_true", help="List TP/FP/FN interactions")
    p.add_argument("--json", action="store_true", help="Print results as JSON")
    p.add_argument("--best-fit", default="none", choices=["none", "single-pair", "greedy-edges", "graph-edges"], help="Only the best fitting family for each basepair is kept")
    p.add_argument("--boundaries", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameter-boundaries.csv"), help="URL/filepath to the boundaries CSV file")
    p.add_argument("--pdbcache", nargs="+", help="Directories to search for PDB files in order to avoid re-downloading")
    return p

if __name__ == "__main__":
    main(parser().parse_args())
