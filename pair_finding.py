import multiprocessing.pool
from typing import Any, Callable, Generator, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple, TypeVar, Union
import os, sys, io, gzip, math, json, numpy as np, re, functools, itertools, csv
import multiprocessing, para_utils
import pdb_utils
import polars as pl
import gemmi

import scipy.spatial.distance

# library for finding close residues

_na_residues = { 'A', 'T', 'G', 'C', 'U', 'DA', 'DT', 'DU', 'DG', 'DC', '5MC', 'BRU', '8OG', '3ZO', 'PSU', 'OMC', '5MU', 'OMU', 'MA6', '2MG', 'UR3', '6MZ', 'G7M','4OC', '2MA','H2U', '4SU', '7MG', '1MG', '1MA' }

def load_structure(pdb: str) -> tuple[gemmi.Structure, pdb_utils.StructureData, str]:
    """Loads the given mmCIF file or PDBID"""
    if "." in pdb:
        match = re.search(r"(?:.*/)?([^/]+)[.](pdb|cif|mmcif)([.](gz|zstd?))?$", pdb, re.IGNORECASE)
        pdbid = match.group(1).lower()
        if not match:
            raise ValueError(f"Invalid file name: {pdb}")
        s = pdb_utils.load_pdb(pdb, pdbid)
        sym = pdb_utils.load_sym_data(pdb,pdbid)
    else: 
        pdbid = pdb.lower()
        s = pdb_utils.load_pdb(None, pdb)
        sym = pdb_utils.load_sym_data(None, pdb)
    if s.name:
        pdbid = s.name.lower()
    else:
        s.name = pdbid

    return s, sym, pdbid


def load_structure_for_basepair(pdb: str, basepair_id: str) -> tuple[gemmi.Structure, pdb_utils.StructureData, str]:
    """
    Loads a structure and returns a new structure containing ONLY the two 
    residues specified in the basepair_id (format: 'chain1_resnum1_chain2_resnum2').
    """
    parts = basepair_id.split('_')
    if len(parts) != 4:
        raise ValueError(f"Invalid basepair_id format: {basepair_id}. Expected 'chain1_res1_chain2_res2'")
    
    target_c1, target_r1, target_c2, target_r2 = parts
    try:
        t1_num, t2_num = int(target_r1), int(target_r2)
    except ValueError:
        raise ValueError(f"Residue numbers must be integers in: {basepair_id}")
    full_s, sym, pdbid = load_structure(pdb)

    filtered_s = gemmi.Structure()
    filtered_s.name = full_s.name
    filtered_s.cell = full_s.cell

    for model in full_s:
        new_model = gemmi.Model(model.name)
        chains_to_add: dict[str, gemmi.Chain] = {}

        for chain in model:
            is_c1 = (chain.name == target_c1)
            is_c2 = (chain.name == target_c2)

            if not (is_c1 or is_c2):
                continue

            for res in chain:
                match1 = (is_c1 and res.seqid.num == t1_num)
                match2 = (is_c2 and res.seqid.num == t2_num)

                if match1 or match2:
                    if chain.name not in chains_to_add:
                        chains_to_add[chain.name] = gemmi.Chain(chain.name)
                    chains_to_add[chain.name].add_residue(res.clone())

        for ch in chains_to_add.values():
            new_model.add_chain(ch)

        if len(new_model) > 0:
            filtered_s.add_model(new_model)

    if len(filtered_s) == 0:
        print(f"Warning: Basepair {basepair_id} not found in structure {pdbid}")

    return filtered_s, sym, pdbid

def find_contacts(s: gemmi.Structure, sym: pdb_utils.StructureData, distance: float) -> pl.DataFrame:
    """Finds contacting residues in structure `s` within `distance` Ångströms."""
    result_dfs: list[pl.DataFrame] = []
    for model_i ,model in enumerate(s):
        ns = gemmi.NeighborSearch(model, s.cell, distance)
        for n_ch, chain in enumerate(model):
            for n_res, residue in enumerate(chain):
                if residue.name.upper() in _na_residues:
                    for n_atom, atom in enumerate(residue):
                        ns.add_atom(atom, n_ch, n_res, n_atom)
        result_dfs.append(_to_contact_df(ns, model, None, distance, s))
        for symop in sym.assembly:
            if not symop.is_identity:
                result_dfs.append(_to_contact_df(ns, model, symop, distance, s))
    # Filter out schema-less empty DataFrames (returned when _to_contact_df finds no contacts)
    # to avoid a schema-mismatch error in pl.concat when mixing empty and non-empty results.
    result_dfs = [df for df in result_dfs if len(df.columns) > 0]
    if not result_dfs:
        return pl.DataFrame()
    return pl.concat(result_dfs)

def collect_all_nucleotide_residues(structure: gemmi.Structure) -> pl.DataFrame:
    """Returns a DataFrame with one row per nucleotide residue across all models."""
    rows = []
    for model in structure:
        model_num = int(model.name) if model.name else 1
        for chain in model:
            for residue in chain:
                if residue.name.upper() not in _na_residues:
                    continue
                rows.append({
                    'model':              model_num,
                    'auth_asym_id':       chain.name,
                    'auth_seq_id':        residue.seqid.num,
                    'auth_comp_id':       residue.name,
                    'pdbx_PDB_ins_code':  residue.seqid.icode if residue.seqid.icode != ' ' else '',
                    'label_asym_id':      residue.subchain  if residue.subchain  else '?',
                    'label_entity_id':    residue.entity_id if residue.entity_id else '?',
                    'label_seq_id':       residue.label_seq if residue.label_seq != -1 else None,
                })
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).with_columns([
        pl.col('model').cast(pl.Int32),
        pl.col('auth_seq_id').cast(pl.Int32),
    ])


def _to_contact_df(ns: gemmi.NeighborSearch, model: gemmi.Model, sym: Optional[pdb_utils.SymmetryOperation], distance: float, structure: gemmi.Structure) -> pl.DataFrame:
    pdbid = structure.name.lower() if structure.name else ""
    _na_residues = {'A', 'T', 'G', 'C', 'U', 'DA', 'DT', 'DU', 'DG', 'DC', '5MC', 'BRU', '8OG', '3ZO', 'PSU', 'OMC', '5MU', 'OMU', 'MA6', '2MG' , 'UR3', '6MZ', 'G7M', '4OC' ,'2MA', 'H2U', '4SU', '7MG', '1MG', '1MA' }
    result = []

    for chain in model:
        for residue in chain:
            if residue.name.upper() not in _na_residues:
                continue

            # Collect all distinct non-blank altlocs present in this residue.
            # Each one is a separate conformer. If none exist, treat as a single '' conformer.
            nonblank_alts = sorted({
                atom.altloc for atom in residue
                if atom.altloc != '\0' and atom.altloc != ' '
            })
            conformers = nonblank_alts if nonblank_alts else ['']

            for alt1 in conformers:
                # found_neighbor_residues is reset per conformer of residue 1
                found_neighbor_residues = set()

                for atom in residue:
                    atom_alt = atom.altloc if atom.altloc not in ('\0', ' ') else ''
                    # Only use atoms belonging to this conformer (specific altloc or shared '')
                    if atom_alt != '' and atom_alt != alt1:
                        continue

                    if sym and not sym.is_identity:
                        pos_array = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                        transformed = sym.rotation @ pos_array + sym.translation
                        search_pos = gemmi.Position(*transformed)
                    else:
                        search_pos = atom.pos

                    neighbors = ns.find_atoms(search_pos, '\0', radius=distance)

                    for neighbor in neighbors:
                        cra = neighbor.to_cra(model)
                        if not cra.residue:
                            continue

                        alt2 = cra.atom.altloc if cra.atom.altloc not in ('\0', ' ') else ''

                        # Skip incompatible altloc pairings:
                        # non-blank altlocs must match; '' is compatible with anything.
                        if alt1 != '' and alt2 != '' and alt1 != alt2:
                            continue

                        res_key = (cra.chain.name, cra.residue.seqid.num, cra.residue.seqid.icode, alt2)

                        if (sym is None and chain.name == cra.chain.name and residue.seqid.num == cra.residue.seqid.num and residue.seqid.icode == cra.residue.seqid.icode):
                            continue

                        if res_key not in found_neighbor_residues:
                            found_neighbor_residues.add(res_key)
                            result.append({
                                # --- Residue 1 ---
                                'label_alt_id1': alt1,
                                'label_comp_id1': residue.name,
                                'label_asym_id1': residue.subchain,
                                'label_entity_id1': residue.entity_id,
                                'label_seq_id1': residue.label_seq if residue.label_seq != -1 else None,
                                'pdbx_PDB_ins_code1': residue.seqid.icode if residue.seqid.icode != ' ' else '',
                                'auth_seq_id1': residue.seqid.num,
                                'auth_comp_id1': residue.name,
                                'auth_asym_id1': chain.name,

                                # --- Residue 2  ---
                                'label_alt_id2': alt2,
                                'label_comp_id2': cra.residue.name,
                                'label_asym_id2': cra.residue.subchain,
                                'label_entity_id2': cra.residue.entity_id,
                                'label_seq_id2': cra.residue.label_seq if cra.residue.label_seq != -1 else None,
                                'pdbx_PDB_ins_code2': cra.residue.seqid.icode if cra.residue.seqid.icode != ' ' else '',
                                'auth_seq_id2': cra.residue.seqid.num,
                                'auth_comp_id2': cra.residue.name,
                                'auth_asym_id2': cra.chain.name,
                            })

    if not result: return pl.DataFrame()

    return pl.DataFrame(result).with_columns(
        pdbid=pl.lit(pdbid, pl.Utf8),
        model=pl.lit(int(model.name) if model.name else 1, pl.Int32),
        # The symop was applied to residue 1's atoms during contact finding (T(res1) ≈ res2),
        # so it belongs on residue 1, not residue 2.
        symmetry_operation1=pl.lit(sym.pdbname if sym and not sym.is_identity else None, pl.Utf8),
        symmetry_operation2=pl.lit(None, pl.Utf8),
        symmetry_id1=pl.lit(sym.id if sym else "1", pl.Utf8),
        symmetry_id2=pl.lit("1", pl.Utf8)
    )

def main_1structure(pdb: str, output_directory: str | None, distance: float):
    try:
        s, sym, pdbid = load_structure(pdb)
    except Exception as e:
        print(f"Error loading {pdb}: {e}")
        return

    result_df = find_contacts(s, sym, distance)

    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)
        csv_path = os.path.join(output_directory, f"{pdbid}.csv")
        result_df.write_csv(csv_path)
        sym_count = len(result_df.filter(pl.col('symmetry_operation2').is_not_null()))
        print(f"Processed {pdbid}: {len(result_df)} contacts ({sym_count} with symmetry operation) -> {csv_path}")
    else:
        return result_df
    

def _main(args, pool: Union[para_utils.MockPool, multiprocessing.pool.Pool]):
    if os.path.exists(args.output):
        if os.listdir(args.output):
            raise ValueError(f"Output directory {args.output} is not empty")
    else:
        os.makedirs(args.output)
    
    inputs = args.inputs
    r = [ pool.apply_async(main_1structure, args=[pdb, args.output, args.distance]) for pdb in inputs ]
    for i, x in enumerate(r):
        x.get()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Finds close residues in the specified PDB structures.')
    parser.add_argument('inputs', nargs='+', type=str, help='PDB files or ids')
    parser.add_argument('--distance', type=float, default=4.0, help='Distance threshold between residue atoms (in Ångströms)')
    parser.add_argument('--output', type=str, help='An empty output directory (will create a parquet file for each PDB structure)', required=True)
    parser.add_argument("--pdbcache", nargs="+", help="Directories to search for PDB files in order to avoid re-downloading. Last directory will be written to, if the structure is not found and has to be downloaded from RCSB. Also can be specified as PDB_CACHE_DIR env variable.")
    parser.add_argument('--threads', type=para_utils.parse_thread_count, default=1, help='Number of threads, 0 for all, 50%% for half, -1 to leave one free, ... Parallelism only affects processing of multiple PDB files.')
    args = parser.parse_args()
    
    pdb_utils.set_pdb_cache_dirs(args.pdbcache)

    if args.threads == 1:
        pool = para_utils.MockPool()
        _main(args, pool)
    else:
        with multiprocessing.Pool(args.threads if args.threads > 0 else max(1, os.cpu_count() + args.threads)) as pool:
            _main(args, pool)