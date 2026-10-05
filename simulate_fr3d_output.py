#!/usr/bin/env python3

import multiprocessing
from multiprocessing.pool import Pool
import pathlib
import re
import typing as ty
import os, sys, io, gzip, math, json, functools, itertools, numpy as np
import polars as pl
from dataclasses import dataclass
import dataclasses
from pair_csv_parse import scan_pair_csvs
import pair_defs as pair_defs
from fr3d_parser import UnitID
from para_utils import MockPool, parse_thread_count

order_sensitive_pairs = [
    pt
    for pt in pair_defs.defined_pair_types()
    if not pt.is_swappable() and not pair_defs.has_symmetrical_definition(pt)
]

def open_file_maybe(f: str | pathlib.Path | ty.IO | ty.TextIO | ty.BinaryIO, mode: str = "r") -> ty.TextIO:
    if isinstance(f, (str, pathlib.Path)):
        return open(f, mode) # type: ignore
    class FakeCx:
        def __init__(self, f):
            self.f = f
        def __enter__(self):
            return self.f
        def __exit__(self, *args):
            pass
    return FakeCx(f) # type: ignore

def write_file(out_file: ty.Any, pdbid_global: ty.Optional[str], df: pl.DataFrame, detailed: bool, only_once: bool, comparison_column: bool, additional_columns: list[str]):
    if pdbid_global is not None:
        assert df["pdbid"].str.to_lowercase().eq(pdbid_global.lower()).all(), f"pdbid mismatch: {pdbid_global} != {df['pdbid'].str.to_lowercase().to_list()}"

    df = df.with_columns(
        _tmp_bases=pl.col("res1").str.to_uppercase().replace(pair_defs.resname_map) + "-" + pl.col("res2").str.to_uppercase().replace(pair_defs.resname_map),
    )
    df = df.with_columns(
        _tmp_pair_type=pl.col("family") + "-" + pl.col("_tmp_bases"),
    )
    df = df.with_columns(
        _tmp_type_is_order_sensitive = pl.col("_tmp_pair_type").str.to_lowercase().is_in([ str(pt).lower() for pt in order_sensitive_pairs ])
    )
    if not detailed:
        # remove near, alternative and bifurcated pairs
        df = df.filter(~pl.col("family").str.starts_with("n")).filter(~pl.col("family").str.ends_with("a")).filter(~pl.col("family").str.ends_with("B"))

    additional_columns_sym = [pl.col("pdbid")] # dummy first column to prevent polars from collapsing the dataframe to 0/1 rows
    additional_columns_asym = [pl.col("pdbid")]
    if comparison_column:
        additional_columns_sym.append(
            pl.when(pl.col("comparison_in_current").not_()).then(pl.lit("MISSING"))
                .when(pl.col("comparison_in_baseline").not_()).then(pl.lit("ADDED"))
                .otherwise(pl.lit("")).alias("comparison_label_column")
        )
    additional_columns_asym.extend(pl.col(col) for col in additional_columns)

    with open_file_maybe(out_file, "w") as f:
        for order_sensitive, pdbid, family, model, chain1, res1, nr1, alt1, ins1, symop1, chain2, res2, nr2, alt2, ins2, symop2, additional_row_sym, additional_row_asym in zip(df["_tmp_type_is_order_sensitive"], df['pdbid'], df["family"], df["model"], df["chain1"], df["res1"], df["nr1"], df["alt1"], df["ins1"], df["symmetry_operation1"], df["chain2"], df["res2"], df["nr2"], df["alt2"], df["ins2"], df["symmetry_operation2"], df.select(*additional_columns_sym).iter_rows(), df.select(*additional_columns_asym).iter_rows()):
            additional_row_sym = [*additional_row_sym][1:]
            additional_row_asym = [*additional_row_asym][1:]
            unit1 = UnitID(pdbid, model, chain1, res1, nr1, "", alt1, ins1, symop1)
            unit2 = UnitID(pdbid, model, chain2, res2, nr2, "", alt2, ins2, symop2)

            m = re.match(r"(?P<n>n)?(?P<cistrans>[ct])(?P<family1>[WHSB])(?P<family2>[WHSB])(?P<alt>[abc1234567890])?", family, re.IGNORECASE)
            assert m is not None, f"Invalid family: {family}"

            if detailed and order_sensitive:
                family = (m.group("n") or '') + m.group("cistrans") + m.group("family1").upper() + m.group("family2").lower() + (m.group("alt") or '')
                family_rev = (m.group("n") or '') + m.group("cistrans") + m.group("family2").lower() + m.group("family1").upper() + (m.group("alt") or '')
            else:
                family = (m.group("n") or '') + m.group("cistrans") + m.group("family1").upper() + m.group("family2").upper() + (m.group("alt") or '')
                family_rev = (m.group("n") or '') + m.group("cistrans") + m.group("family1").upper() + m.group("family2").upper() + (m.group("alt") or '')

            tab = "\t" # fuckin fstrings
            f.write(f"{unit1}\t{family}\t{unit2}{''.join(tab + str(c) for c in additional_row_sym)}{''.join(tab + str(c) for c in additional_row_asym)}\n")
            if not only_once:
                f.write(f"{unit2}\t{family_rev}\t{unit1}{''.join(tab + str(c) for c in additional_row_sym)}{''.join(tab for c in additional_row_asym)}\n")

def main(pool: ty.Union[MockPool, Pool], args):
    df = scan_pair_csvs(args.inputs)

    df = df.sort(pl.col("pdbid").str.to_lowercase(), "model", "chain1", "nr1", "alt1", "ins1", "chain2", "nr2", "alt2", "ins2", "symmetry_operation1", "symmetry_operation2", "family")

    if args.uppercase:
        df = df.with_columns(pdbid=pl.col("pdbid").str.to_lowercase())
    else:
        df = df.with_columns(pdbid=pl.col("pdbid").str.to_lowercase().replace("l", "L"))

    os.makedirs(args.output, exist_ok=True)

    pdbid: str
    procs = []
    for ((pdbid,), group) in df.collect().group_by(["pdbid"]):#type: ignore
        out_file = os.path.join(args.output, pdbid + ("_basepair_detailed.txt" if args.detailed else "_basepair.txt"))
        procs.append(pool.apply_async(write_file, args=[out_file, pdbid, group, args.detailed, args.only_once, args.comparison_column, []]))

    for p in procs:
        p.get()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="""
        Writes files with FR3D-like output from the pair CSV files.
        """,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", "-o", required=True, help="Output directory.")
    parser.add_argument("--detailed", default=False, action="store_true", help="Simulate detailed output with all columns (lowercase second letter, near pairs are not supported)")
    parser.add_argument("--only-once", default=False, action="store_true", help="Do not duplicate pairs")
    parser.add_argument("--uppercase", default=False, action="store_true", help="Uppercase PDB IDs")
    parser.add_argument("--comparison-column", default=False, action="store_true", help="Add a column with information from comparison_in_current/comparision_in_baseline")
    parser.add_argument("--threads", default=None, type=parse_thread_count, help="Parallelism (number, 0 for all, 50%% for half, -1 to leave one free...)")

    args = parser.parse_args()
    multiprocessing.set_start_method("spawn")
    if args.threads == 1:
        main(MockPool(), args)
    else:
        with multiprocessing.Pool(processes=args.threads) as pool:
            main(pool, args)


