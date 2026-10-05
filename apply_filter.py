from types import NoneType
import typing as ty
import os, sys, io, re, gzip, math, json, functools, itertools, numpy as np
import polars as pl
from dataclasses import dataclass
import dataclasses
from pair_csv_parse import scan_pair_csvs
import pair_defs as pair_defs

# New mmCIF-compliant column names
ID_COLS_1 = ["auth_asym_id1", "auth_seq_id1", "label_alt_id1", "pdbx_PDB_ins_code1"]
ID_COLS_2 = ["auth_asym_id2", "auth_seq_id2", "label_alt_id2", "pdbx_PDB_ins_code2"]

def save_output(args, df: pl.LazyFrame):
    df = df.drop(c for c in df.columns if re.match("^_tmp.*$", c))
    if args.output.endswith(".parquet"):
        df.sink_parquet(args.output)
    elif args.output != "/dev/null":
        output_base = args.output.replace(".csv", "")
        df.sink_csv(output_base + ".csv")
        df.sink_parquet(output_base + ".parquet")
    return df

def main(args):
    boundaries = pl.read_csv(args.boundaries, infer_schema_length=10000, dtypes={"subclass": pl.Utf8})
    df = scan_pair_csvs(args.inputs)

    df = apply_filter(df, boundaries, args.null_is_fine, args.accepted_column, args.best_fit)

    # Updated sort with new names
    df = df.sort("pdbid", "model", 
                 "auth_asym_id1", "auth_seq_id1", "label_alt_id1", "pdbx_PDB_ins_code1", 
                 "auth_asym_id2", "auth_seq_id2", "label_alt_id2", "pdbx_PDB_ins_code2", 
                 "symmetry_operation1", "symmetry_operation2", "family")

    save_output(args, df)

def apply_filter(df: pl.LazyFrame, boundaries: pl.DataFrame, null_is_fine: bool, accepted_column: str | None, best_fit: ty.Literal["none", "single-pair", "greedy-edges", "graph-edges"]) -> pl.LazyFrame:
    columns = df.schema.keys()
    if "hb_0_length" not in columns or "C1_C1_yaw1" not in columns:
        raise ValueError("The input data is missing the necessary geometry columns")
    
    has_full_subclass = "full_subclass" in columns
    has_subclass = "subclass" in columns

    # Updated res1/res2 to auth_comp_id1/2
    temp_columns = [
        (pl.col("auth_comp_id1").replace(pair_defs.resname_map).str.to_uppercase() + "-" + 
         pl.col("auth_comp_id2").replace(pair_defs.resname_map).str.to_uppercase()).alias("_tmp_bases"),
        pl.min_horizontal(pl.col("^hb_\\d+_length$").fill_nan(None)).alias("_tmp_min_bond_length"),
    ]

    if has_full_subclass:
        temp_columns.append(pl.col("full_subclass").str.extract(r"_(\d+)$", 1).alias("_tmp_subclass"))
    elif has_subclass:
        temp_columns.append(pl.col("subclass").alias("_tmp_subclass"))
    else:
        temp_columns.append(pl.lit(None, dtype=pl.Utf8).alias("_tmp_subclass"))
    
    df = df.with_columns(temp_columns)

    colname_mapping = {
        "yaw1": "C1_C1_yaw1", "yaw2": "C1_C1_yaw2",
        "pitch1": "C1_C1_pitch1", "pitch2": "C1_C1_pitch2",
        "roll1": "C1_C1_roll1", "roll2": "C1_C1_roll2",
        "min_bond_length": "_tmp_min_bond_length",
    }

    # (boundary_col_name, df_col_name) — boundary name for dict lookup, df name for pl.col()
    checked_columns: list[tuple[str, str]] = [
        (boundary_c, colname_mapping.get(boundary_c, boundary_c))
        for boundary_c in set(col[:-4] for col in boundaries.columns if col.endswith("_min") or col.endswith("_max"))
    ]

    prec_tolerance = 1e-9

    def make_cond(c: str, c_min, c_max, tol: float = prec_tolerance, null_ok: bool = False) -> pl.Expr:
        if c_min is None:
            expr = pl.col(c) <= c_max + tol
        elif c_max is None:
            expr = pl.col(c) >= c_min - tol
        elif c_min > c_max:  # Angular wrap
            expr = pl.col(c).is_between(c_max - tol, c_min + tol).not_()
        else:
            expr = pl.col(c).is_between(c_min - tol, c_max + tol)
        return (expr | pl.col(c).is_null()) if null_ok else expr

    conditions: dict[pair_defs.PairType, pl.Expr] = dict()
    for b in boundaries.iter_rows(named=True):
        pair_type = pair_defs.PairType.from_tuple((b["family"], b["bases"], b["subclass"])).normalize_capitalization()

        other_conds = []
        hb_length_conds = []
        hb_length_relaxed_conds = []
        coplanarity_shift_conds = []
        coplanarity_shift_tight_conds = []

        for boundary_c, c in checked_columns:
            c_min, c_max = b.get(boundary_c + "_min"), b.get(boundary_c + "_max")
            if c_min is None and c_max is None:
                continue

            null_ok = null_is_fine

            if re.match(r"hb_\d+_length$", boundary_c):
                hb_length_conds.append(make_cond(c, c_min, c_max, null_ok=null_ok))
                relaxed_max = (c_max + 0.5) if c_max is not None else None
                hb_length_relaxed_conds.append(make_cond(c, c_min, relaxed_max, null_ok=null_ok))
            elif boundary_c in ("coplanarity_shift1", "coplanarity_shift2"):
                coplanarity_shift_conds.append(make_cond(c, c_min, c_max, null_ok=null_ok))
                coplanarity_shift_tight_conds.append(make_cond(c, -1.1, 1.1, null_ok=null_ok))
            else:
                other_conds.append(make_cond(c, c_min, c_max, null_ok=null_ok))

        all_normal = other_conds + coplanarity_shift_conds + hb_length_conds
        if not all_normal:
            continue

        normal_cond = pl.all_horizontal(all_normal)

        if hb_length_conds:
            all_relaxed = other_conds + coplanarity_shift_tight_conds + hb_length_relaxed_conds
            conditions[pair_type] = normal_cond | pl.all_horizontal(all_relaxed)
        else:
            conditions[pair_type] = normal_cond

    combined_conditions: dict[tuple[str, str], pl.Expr] = {}
    for pt, cond_expr in conditions.items():
        key = (pt.type.upper(), pt.bases_str.upper())
        if key in combined_conditions:
            combined_conditions[key] = combined_conditions[key] | cond_expr
        else:
            combined_conditions[key] = cond_expr

    monster_condition = pl.lit(False)
    for (family, bases), cond_expr in combined_conditions.items():
        match_expr = [
            pl.col("_tmp_bases") == bases,
            pl.col("family").str.to_uppercase() == family,
        ]
        monster_condition = pl.when(pl.all_horizontal(match_expr)).then(cond_expr).otherwise(monster_condition)

    if accepted_column is not None:
        df = df.with_columns(monster_condition.fill_null(False).alias(accepted_column))
    else:
        df = df.filter(monster_condition)

    if best_fit and best_fit != "none":
        df = best_fitting_nucleotides(df, best_fit, accepted_column)

    return df

conflict_resolution_score = (
    (pl.sum_horizontal(pl.col("^hb_\\d+_length$") - 4).fill_null(0)) + 
    pl.col("coplanarity_shift1") + 
    pl.col("coplanarity_shift2")
).alias("conflict_resolution_score")

def filter_variants(df: pl.DataFrame, output_all: bool) -> pl.DataFrame:
    base_pair_id_cols = [
        "pdbid", "model", "auth_asym_id1", "auth_seq_id1", "auth_comp_id1", "label_alt_id1", "pdbx_PDB_ins_code1",
        "auth_asym_id2", "auth_seq_id2", "auth_comp_id2", "label_alt_id2", "pdbx_PDB_ins_code2", "family"
    ]
    df = df.with_columns([
        pl.col("subclass").str.slice(0, 1).alias("_geom_variant"),
        pl.col("subclass").str.slice(1, 1).alias("_dist_variant")
    ])
    
    on_cols = base_pair_id_cols + ["_geom_variant"]
    if not output_all:
        has_strict = df.filter(pl.col("_dist_variant") == "0").select(on_cols).unique().with_columns(pl.lit(True).alias("_has_strict"))
        df = df.join(has_strict, on=on_cols, how="left").filter((pl.col("_dist_variant") == "0") | (pl.col("_has_strict").is_null())).drop("_has_strict")
    else:
        strict_accepted = df.filter((pl.col("_dist_variant") == "0") & (pl.col("accepted") == True)).select(on_cols).unique().with_columns(pl.lit(True).alias("_has_strict_accepted"))
        df = df.join(strict_accepted, on=on_cols, how="left").with_columns(
            pl.when((pl.col("_has_strict_accepted") == True) & (pl.col("_dist_variant") != "0")).then(False).otherwise(pl.col("accepted")).alias("accepted")
        ).drop("_has_strict_accepted")
    return df.drop(["_geom_variant", "_dist_variant"])

def edge_df(df: pl.DataFrame, i: ty.Literal[1, 2]) -> pl.DataFrame:
    # Updated column names in select
    return df.select(
        pl.col("family").str.slice(i, 1),
        pl.col("pdbid"),
        pl.col(f"symmetry_operation{i}"),
        pl.col("model"),
        pl.col(f"auth_asym_id{i}"),
        pl.col(f"auth_seq_id{i}"),
        pl.col(f"pdbx_PDB_ins_code{i}"),
        pl.col(f"label_alt_id{i}"),
    )

def best_fitting_edges_optmatching(df: pl.DataFrame) -> pl.DataFrame:
    try:
        import networkx as nx
        bitmap = np.zeros(len(df), dtype=bool)
        scores = df.select(conflict_resolution_score).to_series().to_numpy()

        graph = nx.Graph()
        for e1, e2, score in zip(edge_df(df, 1).iter_rows(), edge_df(df, 2).iter_rows(), scores):
            weight = max(0.01, 2 - score)
            graph.add_edge(e1, e2, weight=weight)

        # Alternative residue contraction logic updated for mmCIF names
        for res1 in list(graph.nodes):
            neighbors = list(graph.neighbors(res1))
            if len(neighbors) <= 1: continue
            
            # Key for grouping: (edge, pdbid, symop, model, chain, nr, ins) - everything but alt
            no_alt_neighbors = set(n[:-1] for n in neighbors)
            if len(neighbors) == len(no_alt_neighbors): continue

            groups = {}
            for n in neighbors:
                groups.setdefault(n[:-1], []).append(n)

            for g in groups.values():
                if len(g) > 1:
                    for i in range(len(g) - 1):
                        nx.contracted_nodes(graph, g[i], g[i + 1], self_loops=False, copy=False)
                    altcodes = "merged" + "".join(gi[-1] for gi in g)
                    nx.relabel_nodes(graph, {g[-1]: (*g[-1][:-1], altcodes)}, copy=False)

        opt_edges = nx.max_weight_matching(graph)
        opt_set = set(opt_edges) | set((b, a) for a, b in opt_edges)

        for i, (e1, e2) in enumerate(zip(edge_df(df, 1).iter_rows(), edge_df(df, 2).iter_rows())):
            if (e1, e2) in opt_set: bitmap[i] = True

        return df.filter(pl.Series(bitmap))
    except Exception as e:
        print(f"Error in optimal matching: {e}")
        return df

def best_fitting_edges_greedy(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(conflict_resolution_score)
    bitmap = np.zeros(len(df), dtype=bool)
    dedup = set()
    for i, (e1, e2) in enumerate(zip(edge_df(df, 1).iter_rows(), edge_df(df, 2).iter_rows())):
        if e1 not in dedup and e2 not in dedup:
            bitmap[i] = True
            dedup.add(e1); dedup.add(e2)
    return df.filter(pl.Series(bitmap))

def best_fitting_nucleotides(df: pl.LazyFrame, method: str, accepted_column: str|None) -> pl.LazyFrame:
    if accepted_column is not None:
        orig_df = df
        df = df.filter(pl.col(accepted_column))
    
    df = df.sort(conflict_resolution_score)

    # Updated identifier columns
    id_columns = ["family", pl.col("pdbid").str.to_lowercase(), "model", 
                  "auth_asym_id1", "auth_seq_id1", "label_alt_id1", "pdbx_PDB_ins_code1", 
                  "auth_asym_id2", "auth_seq_id2", "label_alt_id2", "pdbx_PDB_ins_code2", 
                  "symmetry_operation1", "symmetry_operation2"]
    
    df = df.group_by(id_columns, maintain_order=True).first()

    if method == "greedy-edges":
        df = df.collect().group_by(["pdbid"]).map_groups(best_fitting_edges_optmatching).lazy()
    elif method == "graph-edges":
        df = df.collect().group_by(["pdbid"]).map_groups(best_fitting_edges_greedy).lazy()
    else:
        df = df.collect().lazy()

    if accepted_column is not None:
        df = orig_df.drop(accepted_column).join(
            df.with_columns(pl.lit(True).alias(accepted_column)).select(id_columns + [accepted_column]),
            on=id_columns, how="left", join_nulls=True
        ).with_columns(pl.col(accepted_column).fill_null(False))

    return df

def parser(parser = None):
    import argparse
    parser = parser or argparse.ArgumentParser(description="Filters RNA basepairs based on geometric boundaries.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--boundaries", type=str, default='./parameter-boundaries.csv')
    parser.add_argument("--null-is-fine", default=True, action="store_true")
    parser.add_argument("--best-fit", default=None, choices=["none", "single-pair", "greedy-edges", "graph-edges"])
    parser.add_argument("--accepted-column", default=None)
    return parser

if __name__ == "__main__":
    main(parser().parse_args())