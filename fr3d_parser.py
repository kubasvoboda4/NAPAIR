

from collections import defaultdict
from dataclasses import dataclass
from multiprocessing.pool import Pool
import os
import re
from typing import Optional, Sequence, TextIO, Union
import polars as pl

import numpy as np

from para_utils import MockPool

def pdb_funny_case(pdbid: str):
    """Lowercase PDBID with uppercase L"""
    return pdbid.lower().replace("l", "L")

@dataclass(frozen=True)
class UnitID:
    """
    Represents https://www.bgsu.edu/research/rna/help/rna-3d-hub-help/unit-ids.html
    Used for parsing FR3D files
    """
    pdbid: str
    """
    PDB ID Code
        From PDBx/mmCIF item: _entry.id
        4 characters, case-insensitive
    """
    model_i: int
    """
    Model Number
        From PDBx/mmCIF item: _atom_site.pdbx_PDB_model_num
        integer, range 1-99
    """
    chain: str
    """
    Model Number
        From PDBx/mmCIF item: _atom_site.pdbx_PDB_model_num
        integer, range 1-99
    """
    residue_base: str
    """
    Residue/Nucleotide/Component Identifier
        From PDBx/mmCIF item: _atom_site.label_comp_id
        1-3 characters, case-insensitive
    """
    residue_component_number: str
    """
    Residue/Nucleotide/Component Number
        From PDBx/mmCIF item: _atom_site.auth_seq_id
        integer, range: -999..9999 (there are negative residue numbers)
    """
    atom_name: str = ""
    """
    Atom Name (Optional, default: blank)
        From PDBx/mmCIF item: _atom_site.label_atom_id
        0-4 characters, case-insensitive
        blank means all atoms
    """
    alternate_id: str = ""
    """
    Alternate ID (Optional, default: blank)
        From PDBx/mmCIF item: _atom_site.label_alt_id
        Default value: blank
        One of ['A', 'B', 'C', '0'], case-insensitive
        This represents alternate coordinates for the model of one or more atoms
    """
    insertion_code: str = ""
    """
    Insertion Code (Optional, default: blank)
        From PDBx/mmCIF item: _atom_site.pdbx_PDB_ins_code
        1 character, case-insensitive
    """
    symmetry_operation: Optional[str] = None
    """
    Symmetry Operation (Optional, default: 1_555)
        As defined in PDBx/mmCIF item: _pdbx_struct_oper_list.name
        5-6 characters, case-insensitive
        For viral icosahedral structures, use “P_” + operator number instead of symmetry operators. For example, 1A34|1|A|VAL|88|||P_1
    """

    @property
    def residue_id(self) -> str:
        """
        Residue ID in form {number}.{insertion_code}, if the insertion_code id is non-empty
        """
        if self.insertion_code:
            return str(self.residue_component_number) + "." + self.insertion_code
        else:
            return str(self.residue_component_number)
        
    @property
    def base_with_alt(self) -> str:
        """
        Base with alternate ID, if the alternate ID is non-empty
        """
        if self.alternate_id:
            return self.residue_base + "." + self.alternate_id
        else:
            return self.residue_base

    @property
    def residue_position(self) -> tuple[str, int, str, str]:
        return (self.pdbid, self.model_i, self.chain, self.residue_id)

    @staticmethod
    def parse(unit_id: str) -> 'UnitID':
        split = unit_id.split('|')
        # more than 5 is ok, sometimes the unit ID is something like 2D34|1|A|DC|5||||8_665, but we don't care about that
        assert len(split) >= 5, f"Invalid unit id {unit_id}"
        pdbid, model_i, chain, nt, ntix = split[:5]
        assert len(pdbid) == 4, f"Invalid pdbid {pdbid}"

        if len(split) > 5:
            atom_name = split[5]
        else:
            atom_name = ""
        
        if len(split) > 6:
            alternate_id = split[6]
        else:
            alternate_id = ""
        
        if len(split) > 7:
            insertion_code = split[7]
        else:
            insertion_code = ""
        
        if len(split) > 8:
            symmetry_operation = split[8]
        else:
            symmetry_operation = None

        return UnitID(pdbid, int(model_i), chain, nt, ntix, atom_name, alternate_id, insertion_code, symmetry_operation)

    def __str__(self) -> str:
        components = [ self.pdbid, str(self.model_i), self.chain, self.residue_base, self.residue_id, self.atom_name, self.alternate_id, self.insertion_code, self.symmetry_operation ]

        while len(components) > 5 and not components[-1]:
            components.pop()

        for i in range(len(components)):
            if components[i] is None:
                components[i] = ""

        return "|".join(components)
    
    def __repr__(self) -> str:
        return str(self)

def read_fr3d_basepairing(file: Union[str, TextIO], pdbid: Optional[str] = None, filter_model = None, filter_chains: Optional[set[str]] = None) -> dict[str, np.ndarray]:
    """
    Reads the fr3d basepairing file into a dictionary of
    * model_i: int - model identifier
    * chain1: string
    * chain2: string
    * res1: string - first nucleotide of the pair
    * res2: string - second nucleotide
    * nr1: int - second nucleotide index (index is 1-based in the whole sequence)
    * nr2: int - second nucleotide index
    * family: string - basepairing family (cWW, tWH, etc.)
    """
    if isinstance(file, str):
        if file.endswith('.gz'):
            import gzip
            with gzip.open(file, 'rt') as f:
                return read_fr3d_basepairing(f, pdbid, filter_model, filter_chains)
        elif file.endswith('.zst'):
            import zstandard
            with zstandard.open(file, 'rt') as f:
                return read_fr3d_basepairing(f, pdbid, filter_model, filter_chains)
        else:
            with open(file, 'rt') as f:
                return read_fr3d_basepairing(f, pdbid, filter_model, filter_chains)

    pairs: dict = {
        "pdbid": [],
        "model": [],
        "chain1": [],
        "chain2": [],
        "res1": [],
        "res2": [],
        "nr1": [],
        "nr2": [],
        "alt1": [], "alt2": [],
        "ins1": [], "ins2": [],
        "symmetry_operation1": [],
        "symmetry_operation2": [],
        "family": [],
    }
    all_models = defaultdict(lambda: 0)
    all_chains = defaultdict(lambda: 0)
    for line in file:
        left_unit_id, basepair_type, right_unit_id, some_number_which_is_always_zero_so_whatetever = line.split()
        left = UnitID.parse(left_unit_id)
        right = UnitID.parse(right_unit_id)
        if pdbid is not None:
            assert left.pdbid == pdbid, f"Invalid pdbid {left.pdbid} in {line}"
            assert right.pdbid == pdbid, f"Invalid pdbid {right.pdbid} in {line}"

        if right.model_i != left.model_i:
            print(f"WARNING: {left} has pairing with different model {right}")

        all_models[left.model_i] += 1
        # print("filter_model", filter_model)
        if filter_model is not None:
            if filter_model == 'first':
                filter_model = left.model_i
            elif left.model_i != filter_model:
                continue
        all_chains["-".join(sorted((left.chain, right.chain)))] += 1
        if filter_chains is not None:
            if left.chain not in filter_chains or right.chain not in filter_chains:
                continue

        pairs["pdbid"] = left.pdbid.lower()
        pairs["model"].append(left.model_i)
        pairs["res1"].append(left.residue_base)
        pairs["res2"].append(right.residue_base)
        pairs["chain1"].append(left.chain)
        pairs["chain2"].append(right.chain)
        pairs["nr1"].append(int(left.residue_component_number))
        pairs["nr2"].append(int(right.residue_component_number))
        pairs["alt1"].append(left.alternate_id)
        pairs["alt2"].append(right.alternate_id)
        pairs["ins1"].append(left.insertion_code)
        pairs["ins2"].append(right.insertion_code)
        pairs["symmetry_operation1"].append(left.symmetry_operation or None)
        pairs["symmetry_operation2"].append(right.symmetry_operation or None)
        pairs["family"].append(basepair_type)

    if filter_model is not None and len(all_models) > 0 and filter_model not in all_models:
        print(f"WARNING: model filter ({filter_model}) filtered out all basepairs in {pdbid}. All models: {dict(sorted(all_models.items()))}")
    if len(all_chains) > 0 and len(pairs["model"]) == 0:
        print(f"NOTE: chain filter ({filter_chains}) filtered out all basepairs in {pdbid}. All chains: {dict(sorted(all_chains.items()))}")

    # pairs["pdbid"] = np.array(pairs["pdbid"], dtype=np.str_)
    # pairs["model"] = np.array(pairs["model"], dtype=np.int32)
    # pairs["res1"] = np.array(pairs["res1"], dtype=np.str_)
    # pairs["res2"] = np.array(pairs["res2"], dtype=np.str_)
    # pairs["chain1"] = np.array(pairs["chain1"], dtype=np.str_)
    # pairs["chain2"] = np.array(pairs["chain2"], dtype=np.str_)
    # pairs["nr1"] = np.array(pairs["nr1"], dtype=np.str_)
    # pairs["nr2"] = np.array(pairs["nr2"], dtype=np.str_)
    # pairs["alt1"] = np.array(pairs["alt1"], dtype=np.str_)
    # pairs["alt2"] = np.array(pairs["alt2"], dtype=np.str_)
    # pairs["ins1"] = np.array(pairs["ins1"], dtype=np.str_)
    # pairs["ins2"] = np.array(pairs["ins2"], dtype=np.str_)
    # pairs["symmetry_operation"] = np.array(pairs["symmetry_operation"], dtype=np.str_)
    # pairs["family"] = np.array(pairs["family"], dtype=np.str_)
    # print(pairs)
    pairs["type"] = pairs["family"]
    return pairs

def find_pairing_files(directory):
    if directory is None:
        return None

    result = dict()
    for f in os.listdir(directory):
        if re.search(r"^[a-zA-Z0-9]{4}_basepair", f):
            pdbid = f.split("_")[0]
            if pdbid in result:
                print("WARNING: duplicate basepairing PDBID", pdbid, ":", f, "and", result[pdbid])
            result[pdbid] = os.path.join(directory, f)
    return result
def _load_frame(file, filter):
    try:
        bps = read_fr3d_basepairing(file)
    except Exception as e:
        raise Exception(f"Error parsing {file}") from e
    if len(bps["model"]) == 0:
        return None
    return pl.DataFrame(bps, schema_overrides={
        "symmetry_operation1": pl.Utf8,
        "symmetry_operation2": pl.Utf8,
        "ins1": pl.Utf8,
        "ins2": pl.Utf8,
        "alt1": pl.Utf8,
        "alt2": pl.Utf8,
    }).filter(filter)

def read_fr3d_files_df(pool: Union[Pool, MockPool], files: Sequence[str], filter=pl.lit(True)) -> pl.DataFrame:
    if pool is None:
        frames = []
        for f in files:
            df = _load_frame(f, filter)
            if df is not None and len(df) > 0:
                frames.append(df)
    else:
        frames = [
            pool.apply_async(_load_frame, args=(f, filter))
            for f in files
        ]
        frames = [ f.get() for f in frames ]
        frames = [ f for f in frames if f is not None ]
    return pl.concat(frames)
