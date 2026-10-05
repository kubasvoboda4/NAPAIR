import gemmi
import os, gzip, io, dataclasses, tempfile
from typing import Any, Callable, Optional, TextIO, TypeVar, Union
import numpy as np

pdb_cache_dirs: list[str] = [ x for x in os.environ.get("PDB_CACHE_DIR", "").split(";") if x.strip() != "" ]

def set_pdb_cache_dirs(dirs: list[str]|None):

    for x in dirs or []:
        pdb_cache_dirs.append(os.path.abspath(x))
    # set the env variable to remember the cache dirs
    os.environ["PDB_CACHE_DIR"] = ';'.join(pdb_cache_dirs)

def _find_in_cache(pdbid):
    """Try to find cif file in the `pdb_cache_dirs`, returns the file path"""
    for cache_dir in pdb_cache_dirs:
        if not os.path.isdir(cache_dir):
            continue
        all_files = os.listdir(cache_dir)
        all_files = { f.lower(): f for f in all_files }
        extensions = [ ".cif.gz", ".cif.zst", ".cif" ]

        for ext in extensions:
            if pdbid.lower() + ext in all_files:
                return os.path.join(cache_dir, all_files[pdbid.lower() + ext])

        subdir = all_files.get(pdbid[:2].lower(), None)
        if subdir is not None:
            subdir = os.path.join(cache_dir, subdir)
            file = next((os.path.join(subdir, pdbfile) for pdbfile in os.listdir(subdir) if pdbfile.lower().startswith(pdbid.lower() + ".")), None)
            if file is not None:
                return file
            
def _get_cache_write_path(pdbid, compression = 'zst'):
    """Returns a file path where to write the downloaded PDB file."""
    if len(pdb_cache_dirs) == 0:
        return None
    pdb_cache_dir = pdb_cache_dirs[-1]
    subdir = os.path.join(pdb_cache_dir, pdbid[:2].lower())
    return os.path.join(subdir, pdbid.lower() + ".cif." + compression)

def _find_or_add_cache(pdbid):
    """Find a cif file in cache, or download it, add it to the cache and return the path."""
    cache_file = _find_in_cache(pdbid)
    if cache_file is not None:
        return cache_file
    else:
        write_path = _get_cache_write_path(pdbid, compression='gz')
        if write_path is None:
            raise Exception("No cache directory")
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        with download_pdb(pdbid) as f:
            write_path_ = write_path + ".part" + str(os.getpid())
            with gzip.open(write_path_, "wt", compresslevel=9) as f2:
                f2.write(f.read())
            os.rename(write_path_, write_path)
        return write_path
        
def download_pdb(pdb_id: str) -> str:
    import urllib.request, tempfile
    url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
    suffix= ".cif.gz"
    response = urllib.request.urlopen(url)
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=f'{pdb_id}_')
    try:
        with os.fdopen(fd, 'wb') as tmp_file:
            tmp_file.write(response.read())
        return temp_path
    except:
        os.unlink(temp_path)
        raise

def open_pdb_file(file: Optional[str], pdbid: Optional[str] = None) -> str:
    if file is None and pdbid is not None:
        if "/" in pdbid or "\\" in pdbid:
            return pdbid
        elif len(pdb_cache_dirs) == 0:
            return download_pdb(pdbid)
        else:
            return _find_or_add_cache(pdbid)
        
    elif isinstance(file,str):
        if not os.path.exists(file):
            raise FileNotFoundError(f"File nit found: {file}")
        return file
    elif file is None:
        raise ValueError("No file specified")
    else:
        fd, temp_path = tempfile.mkstemp(suffix='.cif')
        try:
            with os.fdopen(fd, 'w') as tmp_file:
                content = file.read()
                tmp_file.write(content)
            return temp_path
        except:
            os.unlink(temp_path)
            raise
        
def load_pdb(file:Optional[Union[str | object]], pdbid: Optional[str]= None) -> gemmi.Structure:
    file_path = open_pdb_file(file, pdbid)

    try:
        structure = gemmi.read_structure(file_path)
        if file is None and pdbid is not None and file_path.startswith(tempfile.gettempdir()):
            try:
                os.unlink(file_path)
            except:
                pass
        return structure
    except Exception as e:
        print(f"Error loading {pdbid or file_path}: {e}")
        if file is None and pdbid is not None and file_path.startswith(tempfile.gettempdir()):
            try:
                os.unlink(file_path)
            except:
                pass
        raise

@dataclasses.dataclass
class SymmetryOperation:
    id: str
    pdbname: str
    triplet: str
    rotation: np.ndarray
    translation: np.ndarray

@dataclasses.dataclass
class StructureData:
    assembly: list[SymmetryOperation]

def load_sym_data(file: Optional[Union[str | object]], pdb_id: Optional[str] = None) -> StructureData:
    """
    Extracts some metadata from a mmCIF file. Either file or pdbid must be specified: 
    load_sym_data(None, '1ehz') or load_sym_data('../data/1ehz.cif.gz')
    """
    file_path = open_pdb_file(file, pdb_id)
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                content = f.read()
            if not content.startswith('data_'):
                raise ValueError(f"File does notstart with data_: {content[:100]}")
            doc = gemmi.cif.read_string(content)
        else:
            doc = gemmi.cif.read_file(file_path)
        block = doc.sole_block()
        result = StructureData([])
        oper_list = block.find_mmcif_category('_pdbx_struct_oper_list.')
        
        if oper_list:
            tags = oper_list.tags

            for row in oper_list:
                result.assembly.append(SymmetryOperation(
                    id=str(row['_pdbx_struct_oper_list.id']),
                    pdbname=row['_pdbx_struct_oper_list.name'],
                    triplet=row['_pdbx_struct_oper_list.symmetry_operation'],
                    rotation=np.array([
                        [float(row['_pdbx_struct_oper_list.matrix[1][1]']), 
                         float(row['_pdbx_struct_oper_list.matrix[1][2]']), 
                         float(row['_pdbx_struct_oper_list.matrix[1][3]'])],
                        [float(row['_pdbx_struct_oper_list.matrix[2][1]']), 
                         float(row['_pdbx_struct_oper_list.matrix[2][2]']), 
                         float(row['_pdbx_struct_oper_list.matrix[2][3]'])],
                        [float(row['_pdbx_struct_oper_list.matrix[3][1]']), 
                         float(row['_pdbx_struct_oper_list.matrix[3][2]']), 
                         float(row['_pdbx_struct_oper_list.matrix[3][3]'])],
                    ]),
                    translation=np.array([
                        float(row['_pdbx_struct_oper_list.vector[1]']),
                        float(row['_pdbx_struct_oper_list.vector[2]']),
                        float(row['_pdbx_struct_oper_list.vector[3]'])
                    ])
                ))
            if file is None and pdb_id is not None and file_path.startswith(tempfile.gettempdir()):
                try:
                    os.unlink(file_path)
                except:
                    pass    
        return result
    except Exception as e:
        if file is None and pdb_id is not None and file_path.startswith(tempfile.gettempdir()):
            try:
                os.unlink(file_path)
            except:
                pass
        raise