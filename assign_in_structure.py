import pathlib
import os, sys, io, re, gzip, math, json, functools, itertools, numpy as np, typing as ty, argparse, multiprocessing.pool
import polars as pl
import pdb_utils
import para_utils
import pair_finding
import apply_filter
import pairs
import pair_defs
from simulate_fr3d_output import open_file_maybe, write_file as fr3d_write
import mmcif_parser

# Column Mapping Helper
MMCIF_COLS = {
    'auth_chain': 'auth_asym_id',
    'auth_nr': 'auth_seq_id',
    'res': 'auth_comp_id',
    'alt': 'label_alt_id',
    'ins': 'pdbx_PDB_ins_code',
}

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def _compute_unpaired(pairs_df: pl.DataFrame, all_residues_df: pl.DataFrame) -> pl.DataFrame:
    if len(all_residues_df) == 0:
        return pl.DataFrame()
    if len(pairs_df) == 0:
        return all_residues_df
    paired_keys = pl.concat([
        pairs_df.select([
            'model',
            pl.col('auth_asym_id1').alias('auth_asym_id'),
            pl.col('auth_seq_id1').cast(pl.Int32).alias('auth_seq_id'),
            pl.col('pdbx_PDB_ins_code1').alias('pdbx_PDB_ins_code'),
        ]),
        pairs_df.select([
            'model',
            pl.col('auth_asym_id2').alias('auth_asym_id'),
            pl.col('auth_seq_id2').cast(pl.Int32).alias('auth_seq_id'),
            pl.col('pdbx_PDB_ins_code2').alias('pdbx_PDB_ins_code'),
        ]),
    ]).unique()
    return all_residues_df.join(
        paired_keys,
        on=['model', 'auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code'],
        how='anti'
    )


def format_output(f: ty.Any, df: pl.DataFrame, format: str, header: bool, parameter_columns: ty.List[str] = [], structure_path: str = None, unpaired_df: pl.DataFrame = None):
    if format.endswith('.gz'):
        import gzip
        with gzip.open(f, 'wt', compresslevel=9) as fGz:
            return format_output(fGz, df, format[:-3], header)

    if format.endswith('.zst'):
        import zstandard
        with zstandard.open(f, 'wt') as fZst:
            return format_output(fZst, df, format[:-4], header)
        
    if 'rmsd' in df.columns:
        df = df.with_columns(pl.col('rmsd').round(3))
    if 'knn_metric' in df.columns:
        df = df.with_columns(pl.col('knn_metric').round(3))

    if 'napasco' in df.columns:
        df = df.with_columns([
            pl.col('napasco').round(2).alias('napasco')
        ])

    if 'class' not in df.columns:
        df = df.with_columns([
            (pl.col('family') + '_' + 
             pl.col("auth_comp_id1").str.to_uppercase().replace(pair_defs.resname_map) + '-' + 
             pl.col("auth_comp_id2").str.to_uppercase().replace(pair_defs.resname_map)).alias('class')
        ])
    if 'full_subclass' not in df.columns:
        df = df.with_columns([
            (pl.col('class') + '_' + pl.col('subclass')).alias('full_subclass')
        ])

    # Updated identifier columns for Gemmi/mmCIF
    nt_id_columns = [
        'label_alt_id', 'label_comp_id', 'label_asym_id', 'label_entity_id', 
        'label_seq_id', 'pdbx_PDB_ins_code', 'auth_seq_id', 'auth_comp_id', 
        'auth_asym_id', 'symmetry_operation', 'symmetry_id'
    ]
    id_columns = ['pdbid', 'model', 'family']

    existing_parameters = [col for col in parameter_columns if col in df.columns]
    rmsd_napasco_columns = [c for c in ['napasco', 'rmsd'] if c in df.columns]
    other_validation_columns = [col for col in ['curated_file', 'knn_metric'] if col in df.columns]
    
    handled_columns = set(rmsd_napasco_columns + other_validation_columns)
    other_parameter_columns = [col for col in existing_parameters if col not in handled_columns]
    other_parameter_columns.extend([c for c in df.columns if c.startswith('hb_')])
    
    table_columns = [
        *id_columns, 'class', 'full_subclass',
        *(["accepted"] if "accepted" in df.columns else []),
        *[x + '1' for x in nt_id_columns],
        *[x + '2' for x in nt_id_columns],
        *rmsd_napasco_columns,
        *other_validation_columns,
        *other_parameter_columns
    ]
    json_columns = [
        *id_columns,
        'class',  # Use the column name directly
        *(["accepted"] if "accepted" in df.columns else []),
        pl.struct(pl.col(c + '1').alias(c) for c in nt_id_columns).alias('nt1'),
        pl.struct(pl.col(c + '2').alias(c) for c in nt_id_columns).alias('nt2'),
        pl.struct(pl.col(c) for c in other_parameter_columns).alias('params')
    ]
    
    jsoner = json.encoder.JSONEncoder(ensure_ascii=False, indent=None, separators=(', ', ': '))

    def json_lines(df: pl.DataFrame, include_pdbid: bool = True):
        cols = table_columns if include_pdbid else [c for c in table_columns if c != 'pdbid']
        df = df.select(cols)
        def dict_drop_nulls(d):
            if not isinstance(d, dict):
                return d
            return { k: dict_drop_nulls(v) for k, v in d.items() if v is not None }
        for row in df.iter_rows(named=True):
            yield dict_drop_nulls(row)

    if format == 'csv':
        df.select(table_columns).write_csv(f, include_header=header)
    elif format == 'parquet':
        df.select(table_columns).write_parquet(f) # type:ignore
    elif format == 'arrow':
        df.select(table_columns).write_ipc(f) # type:ignore
    elif format == 'json':
        with open_file_maybe(f, 'wt') as io:
            io.write('{\n')
            for (pdbid,), dfg in df.group_by(['pdbid']):
                io.write(f'  {jsoner.encode(pdbid)}: [\n')
                for i, row in enumerate(json_lines(dfg, include_pdbid=False)):
                    if i > 0:
                        io.write(',\n')
                    io.write('    ')
                    io.write(jsoner.encode(row))
                io.write('\n  ]')
            io.write('\n}\n')
    elif format == 'jsonl' or format == 'jsonnd':
        with open_file_maybe(f, 'wt') as io:
            for row in json_lines(df):
                io.write(jsoner.encode(row))
                io.write('\n')
    elif format == 'fr3d':
        # fr3d writer expects the old short column names
        fr3d_df = df.rename({
            f'{new}{i}': f'{old}{i}'
            for new, old in [('auth_asym_id', 'chain'), ('auth_comp_id', 'res'), ('auth_seq_id', 'nr'), ('label_alt_id', 'alt'), ('pdbx_PDB_ins_code', 'ins')]
            for i in (1, 2)
        })
        fr3d_write(f, None, fr3d_df, detailed=True, only_once=False, comparison_column=False, additional_columns=[])
    elif format == 'mmcif':
        include_validation = 'rmsd' in df.columns or 'validation_dir' in parameter_columns
        mmcif_parser.write_dataframe_to_mmcif(f, df, include_validation, structure_path, unpaired_df)
    else:
        raise ValueError(f'Unknown output format {format}')

def analyze_structure(pdbid: str, min_atom_distance: float, boundaries: pl.DataFrame, best_fit: ty.Literal["none", "single-pair", "greedy-edges", "graph-edges"], output_all: bool, validation_dir: str, basepair_id: str | None = None) -> tuple[str, pl.DataFrame, pl.DataFrame] | None:
    pdb_input = pdbid
    try:
        structure, sym_data, pdbid = pair_finding.load_structure(pdb_input)
    except Exception as e:
        eprint(f"Error loading {pdb_input}: {e}")
        return None

    all_residues_df = pair_finding.collect_all_nucleotide_residues(structure)

    if basepair_id is not None:
        struct, symdata, pdbid = pair_finding.load_structure_for_basepair(pdb_input, basepair_id)
        df = pair_finding.find_contacts(struct, symdata, 200.0)
    else:
        df = pair_finding.find_contacts(structure, sym_data, min_atom_distance)

    if df is None or len(df) == 0:
        return pdbid, pl.DataFrame(), all_residues_df

    df = pairs.override_pair_family(df, pairs.all_familites, boundaries)

    if "subclass" not in df.columns:
        df = df.with_columns(pl.lit("1").alias("subclass"))

    geometry_metrics = [
        pairs.StandardMetrics.Coplanarity,
        pairs.StandardMetrics.YawPitchRoll,
        pairs.StandardMetrics.YawPitchRoll2,
    ]

    _, df, _ = pairs.make_stats_columns(pdbid, df, False, 4, geometry_metrics, structure=structure)

    hb_thresh = 1000 if basepair_id else 4.2
    shift_thresh = 1000 if basepair_id else 2.2

    df = pairs.postfilter_hb(df, hb_thresh)

    df = pairs.postfilter_shift(df, shift_thresh)

    sort_cols = [
        pl.col("pdbid").str.to_lowercase(), "model",
        "auth_asym_id1", "auth_seq_id1", "pdbx_PDB_ins_code1", "label_alt_id1",
        "auth_asym_id2", "auth_seq_id2", "pdbx_PDB_ins_code2", "label_alt_id2",
        "family"
    ]

    df = apply_filter.apply_filter(df.lazy(), boundaries, False, "accepted" if output_all else None, best_fit)\
        .sort(sort_cols)\
        .collect()

    if not output_all:
        df = pairs.remove_duplicate_pairs(df)

    if validation_dir is not None:
        rmsd_metrics = [pairs.get_curated_metric(validation_dir)]
        _, df, _ = pairs.make_stats_columns(pdbid, df, False, 4, rmsd_metrics, structure=structure)

    return pdbid, df, all_residues_df

def _unique_stem(out_dir: str, stem: str, formats: ty.List[str]) -> str:
    """Returns `stem`, or `stem_1`, `stem_2`, ... for the first number where none of the output files (one per format) exist yet, so all formats of one run share the same name"""
    candidate = stem
    i = 0
    while any(os.path.exists(os.path.join(out_dir, f'{candidate}.{fmt}')) for fmt in formats):
        i += 1
        candidate = f'{stem}_{i}'
    return candidate

def _output_name(structure: str, pdbid: str) -> str:
    """Base name for output files: the input file's name for local files (the data_ block of a local file may be a placeholder such as XXXX), otherwise the PDB ID"""
    if not os.path.isfile(structure):
        return pdbid
    name = os.path.basename(structure)
    name = re.sub(r'(\.gz|\.zst)?$', '', name)
    return os.path.splitext(name)[0]

def main_structure(structure: str, boundaries: pl.DataFrame, out: str | None, format: str, best_fit: ty.Literal["none", "single-pair", "greedy-edges", "graph-edges"], output_all: bool, validation_dir: str | None = None, basepair_id: str | None = None):
    """Loads the structure file, performs assignment and writes results"""
    analyzed_struct = analyze_structure(structure, 4.0, boundaries, best_fit, output_all, validation_dir, basepair_id)
    if analyzed_struct is None:
        return
    pdbid, df, all_residues_df = analyzed_struct
    if len(df) == 0:
        eprint(f"No basepairs found for {pdbid}")
        return

    unpaired_df = _compute_unpaired(df, all_residues_df)

    if validation_dir is not None:
        param_columns = ["rmsd", "curated_file","napasco", "knn_metric", 
                        "coplanarity_angle", "coplanarity_shift1", "coplanarity_shift2", 
                        "coplanarity_edge_angle1", "coplanarity_edge_angle2",
                        "C1_C1_yaw1", "C1_C1_pitch1", "C1_C1_roll1", 
                        "C1_C1_yaw2", "C1_C1_pitch2", "C1_C1_roll2"]
    else:
        param_columns = ["coplanarity_angle", "coplanarity_shift1", "coplanarity_shift2", 
                        "coplanarity_edge_angle1", "coplanarity_edge_angle2",
                        "C1_C1_yaw1", "C1_C1_pitch1", "C1_C1_roll1", 
                        "C1_C1_yaw2", "C1_C1_pitch2", "C1_C1_roll2"]

    out_name = _output_name(structure, pdbid)
    stem = f'{out_name}-{basepair_id}' if basepair_id is not None else out_name
    formats = format.split(',')
    stem = _unique_stem(out or '.', stem, formats)
    for fmt in formats:
        file = os.path.join(out or '.', f'{stem}.{fmt}')
        format_output(file, df, fmt, True, param_columns, structure_path=structure, unpaired_df=unpaired_df)

def main2(pool: multiprocessing.pool.Pool | para_utils.MockPool, threads: int, args):
    inputs: list[str] = args.inputs
    out: str = args.out
    boundaries = pl.read_csv(args.boundaries, infer_schema_length=10000, dtypes={"subclass": pl.Utf8})

    os.makedirs(out, exist_ok=True)

    results = []
    for i, structure in enumerate(inputs):
        if args.verbose:
            eprint(f'Processing {structure} ({i+1}/{len(inputs)})')
        results.append(pool.apply_async(main_structure, (structure, boundaries, out, args.format, args.best_fit, args.output_all, args.validation, args.basepair_id)))

    for r in results:
        r.get()

def main(args):
    pdb_utils.set_pdb_cache_dirs(args.pdbcache)

    threads = min(len(args.inputs), args.threads)
    if threads > 1:
        if args.verbose:
            eprint(f'Using {threads} worker processes')
        with multiprocessing.pool.ThreadPool(threads) as pool:
            main2(pool, threads, args)
    else:
        main2(para_utils.MockPool(), 1, args)

def parser(parser = None):
    parser = parser or argparse.ArgumentParser(description="""
        Performs basepair assignment for a given mmCIF structure or PDB ID.
        """)
    parser.add_argument('inputs', nargs='+', type=str, help='Input mmCIF/PDBx file or PDB ID to analyze')
    parser.add_argument('--out', type=str, help='Output directory for results.', required=True)
    parser.add_argument('-f', '--format', type=str, default='csv,parquet,fr3d,json,mmcif', help='Output format(s) to generate. By default, all are selected. Supported formats are csv, parquet, fr3d, json, ndjson. One file per input structure will be created.')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--min-atom-distance', type=float, default=4.0, help='Distance threshold between any residue atoms (in Ångströms)')
    parser.add_argument('--boundaries', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameter-boundaries.csv'), help='URL/filepath to the boundaries CSV file')
    parser.add_argument("--pdbcache", nargs="+", help="Directories to search for PDB files in order to avoid re-downloading. Last directory will be written to, if the structure is not found and has to be downloaded from RCSB. Also can be specified as PDB_CACHE_DIR env variable.")
    parser.add_argument("--best-fit", default='none', type=str, help="Only the best fitting family for each basepair is kept", choices=["none", "single-pair", "greedy-edges", "graph-edges"])
    parser.add_argument("--output-all", action='store_true', help="Output all pairs of nucleotides which are close enough. This can be useful for debugging the parameter boundaries to see why a specific pair was *not* assigned. Basepairs which would be assigned as such will have assigned=true.")
    parser.add_argument('--threads', type=para_utils.parse_thread_count, default=1, help='Number of threads, 0 for all, 50%% for half, -1 to leave one free, ... Parallelism only affects processing of multiple PDB files.')
    parser.add_argument('--validation', nargs='?', const=pairs.DEFAULT_CDF_DIR, default=None, help='Enable RMSD calculation, KNN-based validation scoring, and output curated dataset information. Optionally specify CDF models directory (default: cdf_files/ next to the scripts)')
    parser.add_argument('--basepair_id', type=str, help='Basepair id in format chain1_resnum1_chain2_resnum2')

    return parser

if __name__ == '__main__':
    main(parser().parse_args())