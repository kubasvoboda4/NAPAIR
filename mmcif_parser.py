import gemmi
import polars as pl
from typing import Optional, Dict, Tuple, List
import sys, json, math

# Mapping from family codes to Leontis-Westhof family numbers
LW_FAMILY_NUMBERS = {
    'cWW': '1', 'tWW': '2', 'cWH': '3', 'tWH': '4', 'cWS': '5', 
    'tWS': '6', 'cHH': '7', 'tHH': '8', 'cHS': '9', 'tHS': '10',
    'cSS': '11', 'tSS': '12'
}

# Edge name mappings
EDGE_NAMES = {
    'W': 'Watson-Crick',
    'H': 'Hoogsteen',
    'S': 'Sugar'
}

# Orientation mappings
ORIENTATION_MAP = {
    'c': 'cis',
    't': 'trans'
}

# Software information
SOFTWARE_INFO = {
    'name': 'NAPAIR',
    'version': '1.0',
    'url': '?',
    'bp_params_version': '1.0',
    'details': '"Base pair assignment using geometric parameters"'
}


def parse_family(family: str) -> Tuple[str, str, str, str]:
    """
    Parse family code like 'cWW' into components.
    
    Returns: (orientation, edge1, edge2, lw_family_num)
    """
    if len(family) >= 3:
        orientation = ORIENTATION_MAP.get(family[0], '?')
        edge1 = EDGE_NAMES.get(family[1], '?')
        edge2 = EDGE_NAMES.get(family[2], '?')
        lw_num = LW_FAMILY_NUMBERS.get(family[:3], '?')
        return orientation, edge1, edge2, lw_num
    return '?', '?', '?', '?'

def trimmed_geometric_mean(values: List[float], trim_fraction: float = 0.05) -> Optional[float]:
    """
    Geometric mean after discarding the lowest and highest trim_fraction of values.

    Values <= 0 are dropped first since they have no logarithm.
    """
    positive_vals = sorted(v for v in values if v > 0)
    if not positive_vals:
        return None

    n = len(positive_vals)
    trim_count = int(n * trim_fraction)
    trimmed = positive_vals[trim_count: n - trim_count] if trim_count > 0 else positive_vals
    if not trimmed:
        trimmed = positive_vals

    log_mean = sum(math.log(v) for v in trimmed) / len(trimmed)
    return math.exp(log_mean)


def extract_auth_to_label_mapping_to_file(pdb_input: str, temp_file_path: str) -> bool:
    """
    Use gemmi to load a structure and extract auth-to-label mappings to a temporary file.
    
    Parameters:
    - pdb_input: Path to structure file or PDB ID
    - temp_file_path: Path where to save the mapping data
    
    Returns:
    - True if successful, False otherwise
    """
    try:
        import gemmi
        
        # Load the structure with gemmi
        if pdb_input.endswith('.cif') or pdb_input.endswith('.mmcif'):
            doc = gemmi.cif.read(pdb_input)
            block = doc.sole_block()
        elif pdb_input.endswith('.cif.gz') or pdb_input.endswith('.mmcif.gz'):
            doc = gemmi.cif.read(gemmi.read_gz(pdb_input))
            block = doc.sole_block()
        elif pdb_input.endswith('.pdb') or pdb_input.endswith('.pdb.gz'):
            # PDB files don't have label fields
            return False
        else:
            # Try as PDB ID - fetch mmCIF from PDB
            try:
                pdb_id = pdb_input.upper()
                cif_gz = gemmi.fetch_mmcif(pdb_id)
                doc = gemmi.cif.read(gemmi.read_cif_gz(cif_gz))
                block = doc.sole_block()
            except:
                return False
        
        # Find the atom_site loop
        atom_site_loop = block.find_loop('_atom_site.id')
        if not atom_site_loop:
            return False
        
        # Get column indices
        auth_asym_idx = atom_site_loop.find_tag('_atom_site.auth_asym_id')
        auth_seq_idx = atom_site_loop.find_tag('_atom_site.auth_seq_id')
        
        label_asym_idx = atom_site_loop.find_tag('_atom_site.label_asym_id')
        label_seq_idx = atom_site_loop.find_tag('_atom_site.label_seq_id')
        label_entity_idx = atom_site_loop.find_tag('_atom_site.label_entity_id')
        label_comp_idx = atom_site_loop.find_tag('_atom_site.label_comp_id')
        
        if auth_asym_idx == -1 or auth_seq_idx == -1 or label_asym_idx == -1 or label_seq_idx == -1:
            return False
        
        # Extract unique mappings
        mapping = {}
        
        for row in atom_site_loop:
            auth_asym = row[auth_asym_idx]
            auth_seq = row[auth_seq_idx]
            
            # Create key
            key = f"{auth_asym}:{auth_seq}"
            
            # Skip if already have this mapping
            if key in mapping:
                continue
            
            # Extract label values
            label_data = {
                'label_asym_id': row[label_asym_idx] if label_asym_idx != -1 else auth_asym,
                'label_seq_id': row[label_seq_idx] if label_seq_idx != -1 else auth_seq,
                'label_entity_id': row[label_entity_idx] if label_entity_idx != -1 else '?',
                'label_comp_id': row[label_comp_idx] if label_comp_idx != -1 else '?'
            }
            
            # Clean up values
            for k, v in label_data.items():
                if v == '' or v == '.' or v is None:
                    label_data[k] = '?'
            
            mapping[key] = label_data
        
        # Save to file
        with open(temp_file_path, 'w') as f:
            json.dump(mapping, f)
        
        return True
        
    except Exception as e:
        print(f"Warning: Could not extract auth-to-label mapping: {e}", file=sys.stderr)
        return False


def load_auth_to_label_mapping(temp_file_path: str) -> Optional[Dict[str, Dict[str, str]]]:
    """
    Load auth-to-label mapping from temporary file.
    
    Returns:
    - Dictionary with mapping or None if failed
    """
    try:
        with open(temp_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load auth-to-label mapping from temp file: {e}", file=sys.stderr)
        return None

def dataframe_to_mmcif(df: pl.DataFrame, include_validation: bool = False,
                       structure_path: Optional[str] = None,
                       unpaired_df: Optional[pl.DataFrame] = None) -> str:
    """
    Convert basepair dataframe to mmCIF format using gemmi.
    
    Parameters:
    - df: Polars DataFrame with basepair data
    - include_validation: Whether to include validation loop (when --validation is used)
    - structure_path: Path to the original structure file for extracting label mappings
    
    Returns:
    - mmCIF formatted string
    """
    doc = gemmi.cif.Document()
    pdbid = df['pdbid'][0].upper() if 'pdbid' in df.columns and len(df) > 0 else 'UNKNOWN'
    block = doc.add_new_block(pdbid)

    # Extract auth-to-label mapping if structure path is provided
    auth_to_label_map = None
    if structure_path:
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            if extract_auth_to_label_mapping_to_file(structure_path, tmp_path):
                auth_to_label_map = load_auth_to_label_mapping(tmp_path)
                if auth_to_label_map:
                    print(f"Successfully extracted {len(auth_to_label_map)} auth-to-label mappings", file=sys.stderr)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    unique_pair_columns = ['model', 'auth_asym_id1', 'auth_seq_id1', 'auth_comp_id1', 'pdbx_PDB_ins_code1', 'label_alt_id1', 'symmetry_id1',
                          'auth_asym_id2', 'auth_seq_id2', 'auth_comp_id2', 'pdbx_PDB_ins_code2', 'label_alt_id2', 'symmetry_id2']

    pair_to_id = {}
    base_pair_id = 0
    unique_pairs = []

    for row in df.iter_rows(named=True):
        pair_key = tuple(str(row.get(col, '')) for col in unique_pair_columns)

        if pair_key not in pair_to_id:
            base_pair_id += 1
            pair_to_id[pair_key] = base_pair_id
            unique_pairs.append(row)
    
    num_unique_pairs = len(unique_pairs)

    provenance_loop = block.init_mmcif_loop('_ndb_base_pair_provenance.', [
        'entry_id',
        'software_id', 
        'software_name',
        'software_version',
        'software_url',
        'bp_params_version',
        'details'
    ])
    
    provenance_loop.add_row([
        pdbid,
        '1',
        SOFTWARE_INFO['name'],
        SOFTWARE_INFO['version'],
        SOFTWARE_INFO['url'],
        SOFTWARE_INFO['bp_params_version'],
        SOFTWARE_INFO['details']
    ])

    list_loop = block.init_mmcif_loop('_ndb_base_pair_list.', [
        'id',
        'PDB_model_num',
        'label_asym_id_1',
        'label_entity_id_1',
        'label_seq_id_1',
        'label_comp_id_1',
        'PDB_ins_code_1',
        'label_alt_id_1',
        'struct_oper_id_1',
        'label_asym_id_2',
        'label_entity_id_2',
        'label_seq_id_2',
        'label_comp_id_2',
        'PDB_ins_code_2',
        'label_alt_id_2',
        'struct_oper_id_2',
        'auth_asym_id_1',
        'auth_seq_id_1',
        'auth_asym_id_2',
        'auth_seq_id_2',
    ])

    for i, row in enumerate(unique_pairs):
        bp_id = i + 1

        model = str(row.get('model', '?'))

        # Auth values
        auth_asym_id_1 = row.get('auth_asym_id1', '?')
        auth_seq_id_1 = str(row.get('auth_seq_id1', '?'))
        auth_asym_id_2 = row.get('auth_asym_id2', '?')
        auth_seq_id_2 = str(row.get('auth_seq_id2', '?'))

        # Label values — use DataFrame columns directly, fall back to auth-to-label map, then auth values
        def _label(row, col_label, map_key, map_field, auth_fallback):
            v = row.get(col_label)
            if v and v not in ('?', '.', ''):
                return v
            if auth_to_label_map and map_key in auth_to_label_map:
                return auth_to_label_map[map_key].get(map_field, auth_fallback)
            return auth_fallback

        key1 = f"{auth_asym_id_1}:{auth_seq_id_1}"
        key2 = f"{auth_asym_id_2}:{auth_seq_id_2}"

        asym_id_1   = _label(row, 'label_asym_id1',   key1, 'label_asym_id',   auth_asym_id_1)
        entity_id_1 = _label(row, 'label_entity_id1', key1, 'label_entity_id', '?')
        seq_id_1    = _label(row, 'label_seq_id1',    key1, 'label_seq_id',    auth_seq_id_1)
        comp_id_1   = _label(row, 'label_comp_id1',   key1, 'label_comp_id',   row.get('auth_comp_id1', '?'))

        asym_id_2   = _label(row, 'label_asym_id2',   key2, 'label_asym_id',   auth_asym_id_2)
        entity_id_2 = _label(row, 'label_entity_id2', key2, 'label_entity_id', '?')
        seq_id_2    = _label(row, 'label_seq_id2',    key2, 'label_seq_id',    auth_seq_id_2)
        comp_id_2   = _label(row, 'label_comp_id2',   key2, 'label_comp_id',   row.get('auth_comp_id2', '?'))

        ins1 = row.get('pdbx_PDB_ins_code1', '?')
        ins2 = row.get('pdbx_PDB_ins_code2', '?')
        ins1 = '?' if not ins1 or ins1 in ('', ' ') else ins1
        ins2 = '?' if not ins2 or ins2 in ('', ' ') else ins2

        alt1 = row.get('label_alt_id1', '')
        alt2 = row.get('label_alt_id2', '')
        alt1 = '.' if not alt1 or alt1 in ('', ' ') else alt1
        alt2 = '.' if not alt2 or alt2 in ('', ' ') else alt2

        sym1 = str(row.get('symmetry_id1', '1'))
        sym2 = str(row.get('symmetry_id2', '1'))
        
        list_loop.add_row([
            str(bp_id),
            str(model),
            str(asym_id_1),
            str(entity_id_1),
            str(seq_id_1),
            str(comp_id_1),
            str(ins1),
            str(alt1),
            str(sym1),
            str(asym_id_2),
            str(entity_id_2),
            str(seq_id_2),
            str(comp_id_2),
            str(ins2),
            str(alt2),
            str(sym2),
            str(auth_asym_id_1),
            str(auth_seq_id_1),
            str(auth_asym_id_2),
            str(auth_seq_id_2),
        ])

    # Rest of the function remains the same...
    annotation_loop = block.init_mmcif_loop('_ndb_base_pair_annotation.', [
        'id',
        'base_pair_id',
        'orientation',
        'base_1_edge',
        'base_2_edge',
        'l-w_family_num',
        'l-w_family_name',
        'class',
        'sub_class',
        'details'
    ])

    annotation_id = 0
    
    for row in df.iter_rows(named=True):
        annotation_id += 1

        pair_key = tuple(str(row.get(col, '')) for col in unique_pair_columns)
        bp_id = pair_to_id[pair_key]

        family = row.get('family', '???')
        orientation, edge1, edge2, lw_num = parse_family(family)

        bp_class = row.get('class', f"{family}_{row.get('auth_comp_id1', '?')}-{row.get('auth_comp_id2', '?')}")
        bp_subclass = row.get('full_subclass', row.get('subclass', '1'))
        if not bp_subclass or bp_subclass == '1':
            bp_subclass = f"{bp_class}_1"
        
        annotation_loop.add_row([
            str(annotation_id),
            str(bp_id),
            orientation,
            edge1,
            edge2,
            lw_num,
            family,
            bp_class,
            bp_subclass,
            '?'
        ])

    if include_validation and 'rmsd' in df.columns:
        validation_tags = ['id', 'base_pair_id']
        if 'rmsd' in df.columns:
            validation_tags.append('napair_rmsd')
        if 'napasco' in df.columns:
            validation_tags += ['napasco_metric', 'napasco_annotation']
        if 'curated_file' in df.columns:
            validation_tags.append('nearest_curated_BP')
        validation_tags.append('details')

        validation_loop = block.init_mmcif_loop('_ndb_base_pair_validation.', validation_tags)

        validation_id = 0

        for row in df.iter_rows(named=True):
            validation_id += 1

            pair_key = tuple(str(row.get(col, '')) for col in unique_pair_columns)
            bp_id = pair_to_id[pair_key]

            row_values = [str(validation_id), str(bp_id)]

            if 'rmsd' in df.columns:
                v = row.get('rmsd')
                row_values.append(f"{v:.3f}" if v is not None and v == v else '.')

            if 'napasco' in df.columns:
                v = row.get('napasco')
                if v is None or v != v:  # None or NaN
                    row_values.append('.')
                    row_values.append('not_applicable')
                else:
                    row_values.append(f"{v:.2f}")
                    row_values.append('of_concern' if v < 5 else 'preferred')

            if 'curated_file' in df.columns:
                v = row.get('curated_file')
                row_values.append(str(v) if v is not None else '.')

            validation_loop.add_row(row_values + ['?'])
    

    has_napasco = 'napasco' in df.columns
    unpaired_rows = list(unpaired_df.iter_rows(named=True)) if unpaired_df is not None else []

    def _summary_values(pair_rows, num_pairs, num_unpaired):
        """NAPASCO trimmed geometric mean and annotation counts for a set of pair rows."""
        napasco_values = []
        num_preferred = 0
        num_of_concern = 0
        num_not_applicable = 0

        if has_napasco:
            for row in pair_rows:
                v = row.get('napasco')
                if v is None or v != v:  # None or NaN
                    num_not_applicable += 1
                else:
                    napasco_values.append(v)
                    if v < 5:
                        num_of_concern += 1
                    else:
                        num_preferred += 1

        napasco_val = trimmed_geometric_mean(napasco_values)
        return [
            f"{napasco_val:.2f}" if napasco_val is not None else '.',
            str(num_pairs),
            str(num_preferred) if has_napasco else '.',
            str(num_of_concern) if has_napasco else '.',
            str(num_not_applicable) if has_napasco else '.',
            str(num_unpaired),
        ]

    overall_loop = block.init_mmcif_loop('_ndb_base_pair_overall.', [
        'entry_id',
        'NAPASCO_global',
        'number_of_basepairs',
        'number_of_preferred',
        'number_of_of_concern',
        'number_of_not_applicable',
        'number_of_unpaired',
    ])

    overall_loop.add_row([pdbid] + _summary_values(
        df.iter_rows(named=True), num_unique_pairs, len(unpaired_rows)))

    # Per-model statistics
    pair_rows_by_model = {}
    for row in df.iter_rows(named=True):
        pair_rows_by_model.setdefault(str(row.get('model', '?')), []).append(row)

    unique_pairs_by_model = {}
    for row in unique_pairs:
        model = str(row.get('model', '?'))
        unique_pairs_by_model[model] = unique_pairs_by_model.get(model, 0) + 1

    unpaired_by_model = {}
    for row in unpaired_rows:
        model = str(row.get('model', '1'))
        unpaired_by_model[model] = unpaired_by_model.get(model, 0) + 1

    models = sorted(set(pair_rows_by_model) | set(unpaired_by_model),
                    key=lambda m: (0, int(m), m) if m.isdigit() else (1, 0, m))

    if models:
        model_loop = block.init_mmcif_loop('_ndb_base_pair_overall_model.', [
            'entry_id',
            'PDB_model_num',
            'NAPASCO_global_model',
            'number_of_basepairs',
            'number_of_preferred',
            'number_of_of_concern',
            'number_of_not_applicable',
            'number_of_unpaired',
        ])

        for model in models:
            model_loop.add_row([pdbid, model] + _summary_values(
                pair_rows_by_model.get(model, []),
                unique_pairs_by_model.get(model, 0),
                unpaired_by_model.get(model, 0),
            ))

    if unpaired_df is not None and len(unpaired_df) > 0:
        unpaired_loop = block.init_mmcif_loop('_ndb_base_unpaired_list.', [
            'id',
            'PDB_model_num',
            'label_entity_id',
            'label_asym_id',
            'label_seq_id',
            'label_comp_id',
            'label_alt_id',
            'PDB_ins_code',
            'auth_asym_id',
            'auth_seq_id',
        ])

        for i, row in enumerate(unpaired_rows):
            label_seq = row.get('label_seq_id')
            ins_code  = str(row.get('pdbx_PDB_ins_code') or '')
            unpaired_loop.add_row([
                str(i + 1),
                str(row.get('model', '1')),
                str(row.get('label_entity_id', '?') or '?'),
                str(row.get('label_asym_id',   '?') or '?'),
                str(label_seq) if label_seq is not None else '?',
                str(row.get('auth_comp_id', '?')),
                '.',
                ins_code if ins_code else '?',
                str(row.get('auth_asym_id', '?')),
                str(row.get('auth_seq_id',  '?')),
            ])

    options = gemmi.cif.WriteOptions()
    options.align_pairs = 33
    options.align_loops = 30
    return doc.as_string(options)


def write_dataframe_to_mmcif(f, df: pl.DataFrame, include_validation: bool = False,
                              structure_path: Optional[str] = None,
                              unpaired_df: Optional[pl.DataFrame] = None):
    """
    Write dataframe to mmCIF format using gemmi.
    Parameters:
    - f: File handle or path to write to
    - df: Polars DataFrame with basepair data
    - include_validation: Whether to include validation loop
    - structure_path: Path to the original structure file for extracting atom_site mappings
    - unpaired_df: DataFrame of unpaired nucleotide residues
    """
    cif_string = dataframe_to_mmcif(df, include_validation, structure_path, unpaired_df)
    
    if hasattr(f, 'write'):
        f.write(cif_string)
    else:
        with open(f, 'w') as fh:
            fh.write(cif_string)