import functools
import itertools
from multiprocessing.pool import Pool
import re
from typing import Any, Callable, Generator, Iterator, List, Literal, Optional, Sequence, Tuple, TypeVar, Union
import os, sys, io, gzip, math, json, numpy as np
import polars as pl
from dataclasses import dataclass
import dataclasses
from requests import HTTPError
import multiprocessing
import scipy.spatial
from scipy.spatial.transform import Rotation
from pair_csv_parse import scan_pair_csvs
import pdb_utils
import pair_defs as pair_defs
from para_utils import MockPool, parse_thread_count
from scipy import stats, interpolate
import pickle
import gemmi
pdef = pair_defs

class _Sentinel:
    def __repr__(self):
        return "[sentinel object]"

_sentinel = _Sentinel()
@dataclass
class AltResidue:
    """
    Residue wrapper filtering only atoms with a given altloc code using Gemmi.
    """
    def __init__(self, res: gemmi.Residue, alt: str):
        self.res = res
        self.alt = '\x00' if (alt is None or alt == '' or (isinstance(alt, str) and not alt.strip())) else alt

    @property
    def resname(self) -> str:
        return self.res.name

    def get_atoms(self) -> List[gemmi.Atom]:
        matched_atoms = []
        for atom in self.res:
            a_alt = atom.altloc if atom.altloc != ' ' else '\x00'
            if a_alt == self.alt or a_alt == '\x00':
                matched_atoms.append(atom)
        return matched_atoms

    def get_atom(self, name: str, default: Any = None) -> Optional[gemmi.Atom]:
        target = name.strip()
        for atom in self.res:
            # Strip both to handle any mmCIF whitespace quirks
            atom_name_stripped = atom.name.strip()
            if atom_name_stripped == target:
                # Normalize altloc: convert space/empty to null byte
                a_alt = atom.altloc
                if a_alt == ' ' or a_alt == '':
                    a_alt = '\x00'

                # self.alt is already normalized in __init__
                # Match if altlocs are equal OR atom has no specific altloc
                if a_alt == self.alt or a_alt == '\x00':
                    return atom
        return default
    
def get_coord(atom: gemmi.Atom) -> np.ndarray:
    """Converts Gemmi position to the NumPy float array."""
    return np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)

def try_get_base_atoms(res: AltResidue) -> Optional[Tuple[gemmi.Atom, List[gemmi.Atom]]]:
    # This will now successfully find the atom even if quoted in mmCIF
    c1 = res.get_atom("C1'")

    if c1 is None:

        # Try to iterate atoms
        try:
            atoms_list = list(res.res)
            print(f"  Number of atoms in residue: {len(atoms_list)}", file=sys.stderr)

            # Show first few atoms with C1 in name
            c1_candidates = []
            for atom in atoms_list:
                if "C1" in atom.name:
                    c1_candidates.append((atom.name.strip(), repr(atom.altloc)))
            if c1_candidates:
                print(f"  Found atoms with 'C1' in name: {c1_candidates[:5]}", file=sys.stderr)
            else:
                print(f"  No atoms with 'C1' in name found", file=sys.stderr)

            # Show first 5 atoms
            print(f"  First 5 atoms:", file=sys.stderr)
            for i, atom in enumerate(atoms_list[:5]):
                print(f"    {atom.name.strip()} altloc={repr(atom.altloc)}", file=sys.stderr)
        except Exception as e:
            print(f"  ERROR iterating atoms: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    if c1 is not None:
        all_atoms = res.get_atoms()
        planar_base_atoms = []
        
        for a in all_atoms:
            clean_name = a.name.strip().strip('"')
            
            # Logic: exclude atoms with prime (sugar), exclude phosphates/hydrogens
            if "'" in clean_name or "*" in clean_name:
                continue
            if a.element.name in ["H", "D", "P"]:
                continue
            # Double check for common backbone oxygen/carbon naming (old-style without prime)
            # Note: C2, C4, C5, O4 are intentionally NOT excluded here because they are
            # ring/exocyclic atoms in pyrimidines (C, U, T). filter_atoms() handles any
            # remaining old-style sugar atoms via connectivity.
            if clean_name in ["P", "OP1", "OP2", "OP3", "O5", "O3", "C3"]:
                continue
                
            planar_base_atoms.append(a)
        
        if len(planar_base_atoms) < 6:
            return None
            
        return c1, planar_base_atoms
    return None

def get_base_atoms(res: AltResidue) -> Tuple[gemmi.Atom, List[gemmi.Atom]]:
    """
    Strict version of base atom retrieval.
    """
    x = try_get_base_atoms(res)
    if x is None:
        raise Exception(f"No C1' atom found at residue {res.res.seqid}")
    return x

def bfs(adj_matrix, current):
    assert np.diagonal(adj_matrix).all()
    while True:
        next = adj_matrix @ current
        if np.array_equal(next, current):
            return next
        current = next

class ResideTransformError(Exception):
    pass

def filter_atoms(c1, atoms):
    """
    Returns only atoms connected to the `c1` atom
    """
    c1_vec = get_coord(c1)
    atom_vecs = np.array([get_coord(a) for a in atoms], dtype=np.float64)
    arr = atom_vecs - c1_vec
    c1_bonded = np.linalg.norm(arr, axis=1) <= 1.8
    if not np.any(c1_bonded):
        raise ResideTransformError(f"No bonding atom found at {c1.full_id}")
    adj_matrix = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2) <= 1.6
    connected_component = bfs(adj_matrix, c1_bonded)

    filter = connected_component
    return [ a for a, b in zip(atoms, filter) if b ]

@dataclass
class TranslationThenRotation:
    translation: np.ndarray
    rotation: np.ndarray
    def __post_init__(self):
        assert self.rotation.shape == (3, 3)
        assert self.translation.shape == (3,)
        assert np.isclose(np.linalg.det(self.rotation), 1, atol=1e-4)


@dataclass
class TranslationRotationTranslation:
    translation1: np.ndarray
    rotation: np.ndarray
    translation2: np.ndarray
    def __post_init__(self):
        assert self.rotation.shape == (3, 3)
        assert self.translation1.shape == (3,)
        assert self.translation2.shape == (3,)
        assert np.isclose(np.linalg.det(self.rotation), 1, atol=1e-4)

@dataclass
class ResiduePosition:
    # translation of C1 to origin
    origin: np.ndarray
    # rotation around C1 to align C1'-N bond to x-axis and the plane to X/Y axes
    rotation: np.ndarray
    # orthonormal basis of the base plane (fitted through least squares)
    plane_basis: np.ndarray
    # normal vector of the base plane
    plane_normal_vector: np.ndarray
    # projection matrix to the plane
    projection_matrix: np.ndarray
    # C1' atom coordinates
    c1_pos: Optional[np.ndarray] = None
    # N1/N9 atom coordinates
    n_pos: Optional[np.ndarray] = None

    def __post_init__(self):
        assert self.rotation.shape == (3, 3)
        assert self.plane_basis.shape == (3, 2)

    def plane_projection(self, point: np.ndarray) -> np.ndarray:
        """
        Projects the point onto the base plane
        """
        assert point.shape[-1] == 3
        tpoint = point - self.origin
        return np.matmul(tpoint, self.projection_matrix) + self.origin

    def plane_get_deviation(self, point: np.ndarray) -> float:
        """
        Returns the distance of the point from the plane (in Å), negative if it's on opposite direction of the normal vector
        """
        assert point.shape == (3,)
        distance = math.dist(point, self.plane_projection(point))
        sign = np.sign(np.dot(point - self.origin, self.plane_normal_vector))
        return sign * distance
    
    def get_projection_squish_angle(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Returns the ratio of the distance between the projections and the distance between the points.
        """
        assert point1.shape == (3,) == point2.shape
        proj1 = self.plane_projection(point1)
        proj2 = self.plane_projection(point2)
        dist_cmp = math.dist(proj1, proj2) / math.dist(point1, point2)
        return min(1, max(0, dist_cmp))
    
    def get_relative_line_rotation(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Returns the angle between the line and the plane.
        Negative if the line points in the opposite direction than the normal vector.
        """
        assert point1.shape == (3,) == point2.shape
        line = point2 - point1
        normal_angle = math.degrees(math.acos(np.dot(_linalg_normalize(line), self.plane_normal_vector)))
        return 90 - normal_angle

def fit_plane_magic_svd(atoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns centroid + S (= basis1, basis2, normal)
    https://math.stackexchange.com/a/99317
    """
    assert atoms.shape[1] == 3
    left_singular_vectors, _, _ = np.linalg.svd((atoms - np.mean(atoms, axis=0, keepdims=True)).T)
    assert left_singular_vectors.shape == (3, 3)
    return np.mean(atoms, axis=0), left_singular_vectors

def get_residue_posinfo(res: AltResidue) -> ResiduePosition:
    c1, planar_atoms = get_base_atoms(res)
    # nX is our anchor
    nX = res.get_atom(pair_defs.map_atom_name(res.resname, "N1")) or res.get_atom(pair_defs.map_atom_name(res.resname, "N9"))
    
    # ENSURE COORDINATES ARE FLOATS
    c1_pos = get_coord(c1)
    # Convert all planar atoms to a proper (N, 3) float64 array
    atoms = np.array([get_coord(a) for a in filter_atoms(c1, planar_atoms)], dtype=np.float64)

    # Now your existing math works perfectly
    centroid, rot_matrix1 = fit_plane_magic_svd(atoms)
    plane_basis = rot_matrix1[:, :2]
    normal_v = rot_matrix1[:, 2]
    projection = plane_basis @ plane_basis.T
    
    if nX is None:
        origin = centroid
        rot = rot_matrix1
        nX_pos = None
    else:
        nX_pos = get_coord(nX)
        origin = ((nX_pos - centroid) @ projection) + centroid
        
        # Standardize the rotation system
        c1_c_rot = (c1_pos - origin) @ rot_matrix1
        x_vector = -_linalg_normalize(c1_c_rot)
        rot_matrix2 = np.array([
            [x_vector[0], -x_vector[1], 0],
            [x_vector[1], x_vector[0],  0],
            [ 0,          0,            1]
        ])
        rot = rot_matrix1 @ rot_matrix2

        # Flip check using C8/C6
        ref_name = "C8" if nX.name.strip() == "N9" else "C6"
        left_c = res.get_atom(ref_name)
        if left_c is not None:
            left_pos = get_coord(left_c)
            if ((left_pos - origin) @ rot)[1] > 0:
                rot = rot @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
                normal_v = -normal_v

    return ResiduePosition(origin, rot, plane_basis, normal_v, projection, c1_pos, nX_pos)

def get_residue_posinfo_C1_N(res: AltResidue) -> TranslationThenRotation:
    """
    Orients residue so that the C1-N bond is aligned with the Y-axis, the N1-C2 or N9-C8 bond is towards X axis and remaining is Z
    """
    c1 = res.get_atom("C1'", None)
    if res.get_atom(pair_defs.map_atom_name(res.resname, "N9"), None) is not None:
        n = res.get_atom(pair_defs.map_atom_name(res.resname, "N9"), None)
        c2 = res.get_atom(pair_defs.map_atom_name(res.resname, "C8"), None)
    else:
        n = res.get_atom(pair_defs.map_atom_name(res.resname, "N1"), None)
        c2 = res.get_atom(pair_defs.map_atom_name(res.resname, "C6"), None)
    if c1 is None or n is None or c2 is None:
        raise ResideTransformError(f"Missing atoms in residue {res.res.full_id}")
    # Convert gemmi atoms to numpy arrays
    n_pos = get_coord(n)
    c1_pos = get_coord(c1)
    c2_pos = get_coord(c2)

    translation = -n_pos # N1/N9 is origin
    # y axis is aligned with C1'-N
    y = n_pos - c1_pos
    y /= np.linalg.norm(y)
    # x axis is aligned with N1-C2/N9-C8 (but perpendicular to y)
    x = n_pos - c2_pos
    x -= np.dot(y, x) * y # project X onto Y
    x /= np.linalg.norm(x)
    # z is just perpendicular to x and y
    z = np.cross(x, y)
    z /= np.linalg.norm(z)
    rotation = np.array([x, y, z]).T

    # Optional validation (commented out - too strict for real-world structures)
    # test = transform_residue(res, TranslationThenRotation(translation, rotation))
    # test_n_pos = get_coord(test.get_atom(n.name))
    # test_c1_pos = get_coord(test.get_atom(c1.name))
    # test_c2_pos = get_coord(test.get_atom(c2.name))
    # Validation checks removed - real structures may not match ideal geometry

    return TranslationThenRotation(translation, rotation)

def fit_trans_rot_to_pairs(atom_pairs: Sequence[Tuple[Optional[gemmi.Atom], Optional[gemmi.Atom]]]) -> Optional[TranslationRotationTranslation]:
    """
    Find a translation plus rotation minimizing the RMSD between the atom pairs using Gemmi atoms.
    """
    # Filter out pairs where either atom is None
    valid_pairs = [(a, b) for a, b in atom_pairs if a is not None and b is not None]
    
    if len(valid_pairs) < 3:
        return None

    m1 = np.array([[a.pos.x, a.pos.y, a.pos.z] for a, b in valid_pairs])
    m2 = np.array([[b.pos.x, b.pos.y, b.pos.z] for a, b in valid_pairs])

    trans1 = -np.mean(m1, axis=0)
    m1_centered = m1 + trans1

    center2 = np.mean(m2, axis=0)
    m2_centered = m2 - center2

    try:
        rotation, _rssd, _sensitivity_matrix = scipy.spatial.transform.Rotation.align_vectors(
            m1_centered, 
            m2_centered, 
            return_sensitivity=True
        )
    except ValueError:
        return None

    return TranslationRotationTranslation(trans1, rotation.as_matrix(), center2)

class ResidueInformation:
    def __init__(self, res: AltResidue):
        self.res = res
        self._position = _sentinel
        self._rot_trans = _sentinel
    @property
    def position(self) -> Optional[ResiduePosition]:
        # Don't cache None - always retry if it failed before
        if self._position is _sentinel or self._position is None:
            self._position = try_get_residue_posinfo(self.res)
        return self._position # type: ignore
    @property
    def rot_trans(self) -> Optional[TranslationThenRotation]:
        # Don't cache None - always retry if it failed before
        if self._rot_trans is _sentinel or self._rot_trans is None:
            self._rot_trans = try_x(get_residue_posinfo_C1_N, self.res)
        return self._rot_trans # type: ignore

@dataclass
class PairInformation:
    type: pair_defs.PairType
    resinfo1: ResidueInformation
    resinfo2: ResidueInformation

    @property
    def res1(self) -> AltResidue: return self.resinfo1.res
    @property
    def res2(self) -> AltResidue: return self.resinfo2.res
    @property
    def position1(self) -> Optional[ResiduePosition]: return self.resinfo1.position
    @property
    def position2(self) -> Optional[ResiduePosition]: return self.resinfo2.position
    @property
    def rot_trans1(self) -> Optional[TranslationThenRotation]: return self.resinfo1.rot_trans
    @property
    def rot_trans2(self) -> Optional[TranslationThenRotation]: return self.resinfo2.rot_trans

def calc_pair_information(pair_type: pair_defs.PairType, res1: AltResidue, res2: AltResidue) -> PairInformation:
    p = PairInformation(pair_type, ResidueInformation(res1), ResidueInformation(res2))
    p.resinfo1.position # populate lazy field, it doesn't survive pickling for some reason
    p.resinfo2.position
    p.resinfo1.rot_trans
    p.resinfo2.rot_trans
    return p

def load_pair_information_by_idtuple(pdbid: str, data: list[tuple[Union[tuple[str, str], pair_defs.PairType], tuple]]) -> list[PairInformation]:
    # Load the structure using Gemmi
    structure = pdb_utils.load_pdb(None, pdbid)
    result = []
    
    for pair_type, identifier in data:
        # Unpacking according to the new column schema:
        # pdbid, model, auth_asym_id1, auth_comp_id1, label_seq_id1, auth_seq_id1, pdbx_PDB_ins_code1, label_alt_id1, ...
        (pdbid_loc, model_idx, 
         auth_asym_id1, auth_comp_id1, label_seq_id1, auth_seq_id1, ins1, alt1, 
         auth_asym_id2, auth_comp_id2, label_seq_id2, auth_seq_id2, ins2, alt2) = identifier
        
        assert pdbid_loc.lower() == pdbid.lower()
        pair_type = pair_defs.PairType.from_tuple(pair_type)
        print(f"Loading ideal basepair: {pdbid_loc} Model {model_idx} {auth_asym_id1}:{auth_seq_id1} - {auth_asym_id2}:{auth_seq_id2}")

        # In Gemmi, we access the specific model by index
        # model_idx is usually 1-based in PDB/mmCIF, Gemmi list is 0-based
        model = structure[int(model_idx) - 1]

        # Use a helper to find the GemmiAltResidue (defined in previous step)
        res1_gemmi = find_gemmi_residue(model, auth_asym_id1, auth_seq_id1, ins1)
        res2_gemmi = find_gemmi_residue(model, auth_asym_id2, auth_seq_id2, ins2)
        
        assert res1_gemmi is not None and res2_gemmi is not None, f"Ideal basepairs not found: {identifier}"
        
        # Wrap in our AltResidue handler
        alt_res1 = AltResidue(res1_gemmi, alt1)
        alt_res2 = AltResidue(res2_gemmi, alt2)
        
        assert alt_res1.resname == auth_comp_id1 and alt_res2.resname == auth_comp_id2, \
            f"Residue names don't match: {alt_res1.resname} vs {auth_comp_id1}"

        result.append(calc_pair_information(pair_type, alt_res1, alt_res2))
        
    return result

def find_gemmi_residue(model: gemmi.Model, chain_name: str, res_num: int, icode: str) -> Optional[gemmi.Residue]:
    """Helper to find a residue in a Gemmi model by Author/PDB criteria"""
    if chain_name in model:
        ch = model[chain_name]
        target_seqid = gemmi.SeqId(int(res_num), icode if icode.strip() else ' ')
        res = ch.find_residue(target_seqid)
        return res
    return None

class PairMetric: # "interface"
    columns=None
    def get_columns(self) -> Sequence[str]:
        if self.columns is None:
            raise NotImplementedError()
        return tuple(self.columns)

    def get_values(self, pair: PairInformation) -> Sequence[Optional[float]]:
        raise NotImplementedError()

    def select(self, columns: Sequence[str], rename_as: Optional[Sequence[str]]=None) -> 'PairMetric':
        metric = self
        class SelectPairMetric(PairMetric):
            def get_columns(self) -> Sequence[str]:
                return tuple(rename_as) if rename_as else tuple(columns)
            def get_values(self, pair: PairInformation) -> Sequence[Optional[float]]:
                selection = [ i for i, col in enumerate(metric.get_columns()) if col in columns ]
                vals = list(metric.get_values(pair))
                return tuple([vals[i] for i in selection])
        return SelectPairMetric()
    
    def apply_with(self, metric2: 'PairMetric', fn: Callable[[float, float], float], rename_as: Optional[Sequence[str]]=None) -> 'PairMetric':
        metric = self
        assert len(metric.get_columns()) == len(metric2.get_columns()), f"Cannot apply {fn.__name__} to metrics with different number of columns."

        class ApplyWithPairMetric(PairMetric):
            def get_columns(self) -> Sequence[str]:
                return tuple(rename_as) if rename_as else metric.get_columns()
            def get_values(self, pair: PairInformation) -> Sequence[Optional[float]]:
                return [ (None if a is None or b is None else fn(a, b))
                         for a, b in zip(metric.get_values(pair), metric2.get_values(pair))]

        return ApplyWithPairMetric()
    
    def apply(self, fn: Callable[[float], float], rename_as: Optional[Sequence[str]]=None) -> 'PairMetric':
        metric = self
        class ApplyPairMetric(PairMetric):
            def get_columns(self) -> Sequence[str]:
                return tuple(rename_as) if rename_as else metric.get_columns()
            def get_values(self, pair: PairInformation) -> Sequence[Optional[float]]:
                return [ (None if a is None else fn(a))
                         for a in metric.get_values(pair)]
        return ApplyPairMetric()

    def agg(self, fn: Callable[[list[Optional[float]]], float], rename_as: Optional[str]=None) -> 'PairMetric':
        metric = self
        class AggPairMetric(PairMetric):
            def get_columns(self) -> Sequence[str]:
                return (rename_as,) if rename_as else (metric.get_columns()[0],)
            def get_values(self, pair: PairInformation) -> Sequence[Optional[float]]:
                vals = list(metric.get_values(pair))
                return [fn(vals)]
        return AggPairMetric()

    @staticmethod
    def concat(metrics: list['PairMetric']) -> 'PairMetric':
        class ConcatPairMetric(PairMetric):
            def get_columns(self) -> Sequence[str]:
                return list(itertools.chain(*[m.get_columns() for m in metrics]))
            def get_values(self, pair: PairInformation) -> Sequence[Optional[float]]:
                return list(itertools.chain(*[m.get_values(pair) for m in metrics]))
        return ConcatPairMetric()

def Rx(θ): return np.array([ [1, 0, 0], [0, np.cos(θ), -np.sin(θ)], [0, np.sin(θ), np.cos(θ) ]])
def Rz(θ): return np.array([[ np.cos(θ), -np.sin(θ), 0],[ np.sin(θ), np.cos(θ), 0],[ 0, 0, 1] ])
def Ry(θ): return np.array([[ np.cos(θ), 0, np.sin(θ)],[ 0, 1, 0],[ -np.sin(θ), 0, np.cos(θ)]])

def get_C1_N_yaw_pitch_roll(rot1: np.ndarray, rot2: np.ndarray) -> Tuple[float, float, float]:
    matrix = rot1.T @ rot2
    # https://stackoverflow.com/a/37558238
    # yaw = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    # pitch = math.degrees(math.atan2(-matrix[2, 0], math.sqrt(matrix[2, 1]**2 + matrix[2, 2]**2)))
    # roll = math.degrees(math.atan2(matrix[2, 1], matrix[2, 2]))

    # this function probably the intuitive representation, as it corresponds to:
    # In [57]: Rotation.from_matrix(Rz(0.33) @ Ry(0.44) @ Rx(0.55)).as_euler("ZYX")
    # Out[57]: array([0.33, 0.44, 0.55])
    yaw, pitch, roll = Rotation.from_matrix(matrix).as_euler("ZYX", degrees=True)

    assert np.allclose(matrix, Rz(math.radians(yaw)) @ Ry(math.radians(pitch)) @ Rx(math.radians(roll)), atol=1e-5)
    return yaw, pitch, roll
def get_C1_N_euler_angles(rot1: np.ndarray, rot2: np.ndarray) -> Tuple[float, float, float]:
    matrix = rot1.T @ rot2
    # https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Rotation_matrix_%E2%86%92_Euler_angles_(z-x-z_extrinsic)
    # eu1 = math.degrees(math.atan2(matrix[2, 0], matrix[2, 1]))
    # eu2 = math.degrees(math.acos(matrix[2, 2]))
    # eu3 = -math.degrees(math.atan2(matrix[0, 2], matrix[1, 2]))
    eu1, eu2, eu3 = Rotation.from_matrix(matrix).as_euler("zxz", degrees=True)
    return eu1, eu2, eu3

T = TypeVar('T')
def try_x(f: Callable[[AltResidue], T], res: AltResidue, warning = None) -> Optional[T]:
    try:
        return f(res)
    except ResideTransformError as e:
        if warning is not None:
            warning(str(e))
        else:
            print(f"WARNING(residue = {res.res.full_id} {res.alt}): {e}")
        return None

def try_get_residue_posinfo(res: AltResidue, warning = None) -> Optional[ResiduePosition]:
    return try_x(get_residue_posinfo, res, warning)

TResidue = TypeVar('TResidue', bound=Union[AltResidue, gemmi.Residue])

def transform_residue(r: TResidue, t: Union[TranslationThenRotation, TranslationRotationTranslation, ResiduePosition]) -> TResidue:
    """
    Applies spatial transformations to a Gemmi Residue or AltResidue wrapper.
    """
    if isinstance(r, AltResidue):
        # Transform the underlying gemmi.Residue and return a new wrapper
        return AltResidue(transform_residue(r.res, t), r.alt)

    # 1. Create a deep copy to avoid modifying the original structure
    r_copy = r.clone() 

    # 2. Extract rotation and translations based on the type
    if isinstance(t, TranslationThenRotation):
        # Order: T -> R
        _apply_transformation(r_copy, t.translation, t.rotation, None)
    elif isinstance(t, ResiduePosition):
        # Order: T (origin) -> R
        _apply_transformation(r_copy, t.origin, t.rotation, None)
    elif isinstance(t, TranslationRotationTranslation):
        # Order: T1 -> R -> T2
        _apply_transformation(r_copy, t.translation1, t.rotation, t.translation2)
        
    return r_copy

def _apply_transformation(res: gemmi.Residue, t1: np.ndarray, rot: np.ndarray, t2: Optional[np.ndarray] = None):
    """
    Internal helper to apply matrices to Gemmi atom positions.
    Calculation: pos = (rot @ (pos + t1)) + t2
    """
    for atom in res:
        # Convert gemmi.Position to numpy array
        pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
        
        # Apply Translation 1
        pos += t1
        
        # Apply Rotation
        pos = rot @ pos
        
        # Apply Translation 2 (if exists)
        if t2 is not None:
            pos += t2
            
        # Update Gemmi atom position
        atom.pos = gemmi.Position(pos[0], pos[1], pos[2])

def to_float32_df(df: pl.DataFrame) -> pl.DataFrame:
    """Convert float64 columns to float32"""
    return df.with_columns([
        pl.col(c).cast(pl.Float32).alias(c) for c in df.columns if df[c].dtype == pl.Float64
    ])

def _linalg_norm(v) -> float:
    return math.sqrt(v.dot(v))

def pos_to_array(pos: Union[gemmi.Position, gemmi.Atom]) -> np.ndarray:
    """Safely converts Gemmi position to a float64 numpy array."""
    if hasattr(pos, 'pos'): # It's an Atom with .pos attribute
        return np.array([pos.pos.x, pos.pos.y, pos.pos.z], dtype=np.float64)
    return np.array([pos.x, pos.y, pos.z], dtype=np.float64)

def _linalg_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v

def maybe_agg(agg, array):
    """
    Apply a function to a list if it's not empty
    """
    if len(array) == 0:
        return None
    return agg(array)

def abs_min(array):
    """
    Finds the element closest to zero
    """
    ix = np.argmin(np.abs(array))
    return array[ix]

def get_edge_atoms(res: AltResidue, edge_name: str | Literal['W', 'H', 'S']) -> List[gemmi.Atom]:
    """
    Returns the residue atoms forming a given edge using Gemmi.
    """
    # Assuming pair_defs and resname_map are available in your environment
    res_name = resname_map.get(res.resname, res.resname)
    edge_atom_names = pair_defs.base_edges.get(res_name, {}).get(edge_name, [])
    
    # Use the get_atom method from our GemmiAltResidue wrapper
    return [a for aname in edge_atom_names if (a := res.get_atom(pair_defs.map_atom_name(res.resname, aname))) is not None]

def get_edge_edges(edge: List[gemmi.Atom]) -> Optional[Tuple[gemmi.Atom, gemmi.Atom]]:
    """
    Returns the first and last atom of an edge. 
    Prioritizes polar atoms (N, O, etc.) over carbons and hydrogens.
    """
    if not edge or len(edge) < 2:
        return None
    
    # Gemmi atom.element.name returns the symbol (e.g., 'N', 'O', 'C')
    polar_atoms = [a for a in edge if a.element.name not in ("H", "D", "C")]
    
    if len(polar_atoms) < 2:
        # Fallback to the first and last atoms of the list if not enough polar atoms
        return (edge[0], edge[-1])
        
    return (polar_atoms[0], polar_atoms[-1])

class CoplanarityEdgeMetrics(PairMetric):
    columns = [ "coplanarity_angle", "coplanarity_shift1", "coplanarity_shift2", "coplanarity_edge_angle1", "coplanarity_edge_angle2" ]
    def get_values(self, pair: PairInformation) -> Sequence[Optional[float]]:
        p1, p2 = pair.position1, pair.position2
        if p1 is None or p2 is None:
            return [ None ] * len(self.columns)

        edge1 = get_edge_atoms(pair.res1, pair.type.edge1_name)
        edge1_edges = get_edge_edges(edge1)
        edge2 = get_edge_atoms(pair.res2, pair.type.edge2_name)
        edge2_edges = get_edge_edges(edge2)

        return [
            math.degrees(np.arccos(np.dot(p1.plane_normal_vector, p2.plane_normal_vector))), # coplanarity_angle
            maybe_agg(abs_min, [p2.plane_get_deviation(get_coord(a)) for a in edge1]), # coplanarity_shift1
            maybe_agg(abs_min, [p1.plane_get_deviation(get_coord(a)) for a in edge2]), # coplanarity_shift2
            p2.get_relative_line_rotation(*(get_coord(a) for a in edge1_edges)) if edge1_edges is not None else None, # coplanarity_edge_angle1
            p1.get_relative_line_rotation(*(get_coord(a) for a in edge2_edges)) if edge2_edges is not None else None, # coplanarity_edge_angle2
        ]
    
class IsostericityCoreMetrics(PairMetric):
    """
    C1'-C1' distance and the angle of the two C1'-N bonds, as the two numbers reported in the LSW2002 paper
    """
    columns = [ "C1_C1_distance", "C1_C1_total_angle" ]
    def get_values(self, pair: PairInformation) -> Sequence[Optional[float]]:
        pos1, pos2 = pair.position1, pair.position2
        if pos1 is None or pos2 is None or pos1.c1_pos is None or pos2.c1_pos is None:
            return [ None ] * len(self.columns)

        distance = math.dist(pos1.c1_pos, pos2.c1_pos)
        if pos1.n_pos is None or pos2.n_pos is None:
            total_angle = None
        else:
            total_angle = math.degrees(np.arccos(np.dot(_linalg_normalize(pos1.n_pos - pos1.c1_pos), _linalg_normalize(pos2.c1_pos - pos2.n_pos))))
        return [ distance, total_angle ]
    
class EulerAngleMetrics(PairMetric):
    def __init__(self, col_names: Sequence[str], function: Callable[[np.ndarray, np.ndarray], tuple[float, float, float]], reference: Literal[1, 2]) -> None:
        self.columns = [ "C1_C1_" + c for c in col_names ]
        self.function = function
        self.reference = reference

    def get_values(self, pair: PairInformation) -> Sequence[Optional[float]]:
        rot_trans1, rot_trans2 = pair.rot_trans1, pair.rot_trans2
        if rot_trans1 is None or rot_trans2 is None:
            return [ None ] * len(self.columns)
        flip = np.diag([-1, -1, 1])
        if self.reference == 2:
            return self.function(rot_trans2.rotation, rot_trans1.rotation @ flip)
        else:
            return self.function(rot_trans1.rotation, rot_trans2.rotation @ flip)

class TranslationMetrics(PairMetric):
    """
    x, y, z coordinates in the other residue C1' - N1/N9 coordinate system
    """
    def __init__(self, reference: Literal[1, 2]):
        self.reference = reference
        self.columns = np.char.add(["x", "y", "z"], str(reference))
    def get_values(self, pair: PairInformation) -> Sequence[float | None]:
        ref, other = (pair.rot_trans1, pair.rot_trans2) if self.reference == 1 else (pair.rot_trans2, pair.rot_trans1)
        if ref is None or other is None:
            return [ None ] * 3
        return list(ref.rotation.T @ (other.translation - ref.translation))
    
# Resolved relative to this file, so validation works regardless of the current working directory
DEFAULT_CDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cdf_files")
DEFAULT_CURATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "curated")

class CuratedRMSDMetric(PairMetric):
    """
    Calculate RMSD between the observed base pair and curated, using only specific atoms:
    - For purines: C1', N9, C4, C8
    - For pyrimidines: C1', N1, C2, C6
    
    Also calculates KNN-based CDF scores using pre-computed CDF models.
    Uses enhanced CDF models with confidence intervals when available.
    """
    results_cache = {}

    def __init__(self, cdf_models_dir=DEFAULT_CDF_DIR, use_knn_scoring=True, k_neighbors=5, curated_dir=DEFAULT_CURATED_DIR):
        self.columns = ["rmsd", "curated_file"]
        if use_knn_scoring:
            self.columns.extend(["napasco", "knn_metric"])
        
        self.curated_cache = {}
        self._class_files_cache = {}  # bp_class -> [(file name, path)]
        self._reference_cache = {}    # curated path -> prepared atoms/coords, or None if unusable
        self._pack = None             # lazily opened packed curated library (see curated_pack.py), False if unavailable
        self._packed_classes = {}     # bp_class -> list of same-shape reference groups
        self.use_knn_scoring = use_knn_scoring
        self.k_neighbors = k_neighbors
        self.cdf_models_dir = cdf_models_dir
        self.curated_dir = curated_dir
        self.pack_path = os.path.normpath(curated_dir) + ".npz"
        self.cdf_scorers = {}

        if not os.path.isdir(curated_dir):
            print(f"Warning: curated basepairs directory {curated_dir} not found, validation columns will be empty", file=sys.stderr)
        
        # Define the specific atoms we want to use
        self.purine_atoms = ["C1'", "N9", "C4", "C8"]
        self.pyrimidine_atoms = ["C1'", "N1", "C2", "C6"]
        
        # Load CDF scorers if KNN scoring is enabled
        if use_knn_scoring:
            self.load_cdf_scorers()
    
    def get_columns(self) -> Sequence[str]:
        return self.columns
    
    def load_cdf_scorers(self):
        """Load pre-computed enhanced CDF scorers from pickle files"""
        import glob
        
        if not os.path.exists(self.cdf_models_dir):
            print(f"Warning: CDF models directory {self.cdf_models_dir} not found")
            return
        
        # Only look for enhanced CDF models
        pattern = os.path.join(self.cdf_models_dir, "*_enhanced_cdf.pkl")
        cdf_files = glob.glob(pattern)
        
        if not cdf_files:
            print(f"Warning: No enhanced CDF model files found in {self.cdf_models_dir}")
            return
        
        for file_path in cdf_files:
            filename = os.path.basename(file_path)
            class_name = filename.replace('_enhanced_cdf.pkl', '')
            
            try:
                with open(file_path, 'rb') as f:
                    self.cdf_scorers[class_name] = pickle.load(f)
            except Exception as e:
                print(f"Error loading CDF scorer from {file_path}: {e}")
    
    def calculate_sqrt_sum_reciprocal_squared(self, rmsd_values):
        """Calculate the square root of sum of reciprocals of squared RMSD values"""
        valid_rmsds = [rmsd for rmsd in rmsd_values if rmsd != float('inf') and rmsd > 0]
        
        if not valid_rmsds:
            return 0.0
        
        sum_squared= sum(rmsd**2 for rmsd in valid_rmsds)
        return 1/np.sqrt(sum_squared)
    def get_cdf_score(self, bp_class, knn_metric_value):
        """
        Get enhanced CDF score with confidence interval weighting.
        Formula: napasco[5nn] = CDF[5nn] x (1 - width_of_confidence_interval[5nn])
        """
        if bp_class not in self.cdf_scorers or knn_metric_value is None:
            return None
        
        try:
            scorer_data = self.cdf_scorers[bp_class]
            
            # Check if this is an enhanced model with confidence intervals
            if all(key in scorer_data for key in ['cdf_function', 'lower_ci_function', 'upper_ci_function']):
                # Get CDF value and confidence intervals
                cdf_value = float(scorer_data['cdf_function'](knn_metric_value))
                ci_lower = float(scorer_data['lower_ci_function'](knn_metric_value))
                ci_upper = float(scorer_data['upper_ci_function'](knn_metric_value))
                
                # Ensure values are within valid bounds
                cdf_value = min(1.0, max(0.0, cdf_value))
                ci_lower = min(1.0, max(0.0, ci_lower))
                ci_upper = min(1.0, max(0.0, ci_upper))
                
                # Calculate confidence interval width
                ci_width = ci_upper - ci_lower
                
                # Apply your formula napasco[5nn] = CDF[5nn] × (1 - width_of_confidence_interval[5nn])
                confidence_factor = (1.0 - ci_width)
                enhanced_napasco = cdf_value * confidence_factor
                
                # Scale to percentage (0-100)
                napasco = enhanced_napasco * 100
                
                return napasco
            else:
                print(f"Warning: Model for {bp_class} missing confidence interval data")
                return None
                
        except Exception as e:
            print(f"Error getting CDF score for {bp_class}: {e}")
            return None
      
    def load_curated(self, curated_path):
        """Load a curated structure with caching, parsing CIF directly."""
        if curated_path in self.curated_cache:
            return self.curated_cache[curated_path]

        # Lightweight atom/residue objects — only need .pos.x/y/z and .get_atom()
        class _Pos:
            __slots__ = ('x', 'y', 'z')
            def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z

        class _Atom:
            __slots__ = ('name', 'pos')
            def __init__(self, name, x, y, z):
                self.name = name
                self.pos = _Pos(x, y, z)

        class _Residue:
            def __init__(self, resname: str):
                self._atoms = {}
                self.resname = resname
            def get_atom(self, name, default=None):
                return self._atoms.get(name.strip(), default)

        doc = gemmi.cif.read(curated_path)
        block = doc.sole_block()
        table = block.find('_atom_site.', [
            'label_atom_id', 'label_comp_id', 'label_asym_id', 'label_seq_id',
            'Cartn_x', 'Cartn_y', 'Cartn_z',
        ])

        residues_ordered = []
        residues_map = {}
        for row in table:
            atom_name = row[0].strip()
            comp_id   = row[1].strip()
            asym_id   = row[2].strip()
            seq_id    = row[3].strip()
            try:
                x, y, z = float(row[4]), float(row[5]), float(row[6])
            except (ValueError, TypeError):
                continue
            key = (asym_id, seq_id)
            if key not in residues_map:
                res = _Residue(comp_id)
                residues_map[key] = res
                residues_ordered.append(res)
            residues_map[key]._atoms[atom_name] = _Atom(atom_name, x, y, z)

        nucleotides = [r for r in residues_ordered if r.get_atom("C1'") is not None][:2]
        self.curated_cache[curated_path] = (None, nucleotides)
        return self.curated_cache[curated_path]
    
    def get_specific_atoms(self, residue: AltResidue):
        """Get only the specific atoms we want to use for RMSD calculation"""
        # Determine if purine or pyrimidine
        is_purine_res = residue.get_atom(pair_defs.map_atom_name(residue.resname, "N9")) is not None
        atom_list = self.purine_atoms if is_purine_res else self.pyrimidine_atoms
        atoms = []
        for atom_name in atom_list:
            atom = residue.get_atom(pair_defs.map_atom_name(residue.resname, atom_name))
            if atom is not None:
                atoms.append(atom)

        return atoms, "purine" if is_purine_res else "pyrimidine"
    
    def calculate_basepair_rmsd(self, cur1_atoms, cur2_atoms, r1_atoms, r2_atoms):
        """
        Calculate the RMSD between two base pairs by:
        1. Superimposing base 1, calculating RMSD for base 2
        2. Superimposing base 2, calculating RMSD for base 1
        3. Averaging the two RMSD values
        """
        if len(cur1_atoms) == 0 or len(cur2_atoms) == 0 or len(r1_atoms) == 0 or len(r2_atoms) == 0:
            return float('inf')

        return self.calculate_basepair_rmsd_coords(
            self.atoms_to_coords(cur1_atoms), self.atoms_to_coords(cur2_atoms),
            self.atoms_to_coords(r1_atoms), self.atoms_to_coords(r2_atoms))

    @staticmethod
    def atoms_to_coords(atoms):
        return np.array([[a.pos.x, a.pos.y, a.pos.z] for a in atoms], dtype=np.float64)

    @staticmethod
    def calculate_basepair_rmsd_coords(cur1_coords, cur2_coords, r1_coords, r2_coords):
        """Same as calculate_basepair_rmsd, on precomputed (n, 3) coordinate arrays (all four non-empty)."""
        def kabsch(fixed, moving):
            """Kabsch algorithm: find R, t such that moving @ R + t ≈ fixed (row-vector convention)."""
            fc = fixed.mean(axis=0)
            mc = moving.mean(axis=0)
            H = (moving - mc).T @ (fixed - fc)
            U, _, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            R = U @ np.diag([1.0, 1.0, d]) @ Vt
            t = fc - mc @ R
            return R, t

        # 1. Superimpose base 1 (align r1 to cur1), apply transform to r2
        R, t = kabsch(cur1_coords, r1_coords)
        r2_aligned = r2_coords @ R + t
        diff = cur2_coords - r2_aligned
        rmsd1 = np.sqrt(np.sum(diff * diff) / len(cur2_coords))

        # 2. Superimpose base 2 (align r2 to cur2), apply transform to r1
        R, t = kabsch(cur2_coords, r2_coords)
        r1_aligned = r1_coords @ R + t
        diff = cur1_coords - r1_aligned
        rmsd2 = np.sqrt(np.sum(diff * diff) / len(cur1_coords))

        # 3. Average the RMSDs
        return (rmsd1 + rmsd2) / 2

    def get_class_files(self, bp_class):
        """Curated .cif files of a bp class as (name, path) pairs; the directory is listed once per class."""
        files = self._class_files_cache.get(bp_class)
        if files is None:
            class_dir = os.path.join(self.curated_dir, bp_class)
            names = [f for f in os.listdir(class_dir) if f.endswith('.cif')] if os.path.isdir(class_dir) else []
            files = self._class_files_cache[bp_class] = [(f, os.path.join(class_dir, f)) for f in names]
        return files

    def get_reference(self, curated_path):
        """Atoms, base types and coordinates of a curated pair, resolved once per file. None if it cannot be used."""
        if curated_path not in self._reference_cache:
            ref = None
            try:
                _, residues = self.load_curated(curated_path)
                if len(residues) >= 2:
                    cur1_atoms, cur1_type = self.get_specific_atoms(residues[0])
                    cur2_atoms, cur2_type = self.get_specific_atoms(residues[1])
                    if cur1_atoms and cur2_atoms:
                        ref = (cur1_type, cur2_type, self.atoms_to_coords(cur1_atoms), self.atoms_to_coords(cur2_atoms))
            except Exception as e:
                print(f"Error processing curated {curated_path}: {e}")
            self._reference_cache[curated_path] = ref
        return self._reference_cache[curated_path]

    def get_packed_class(self, bp_class):
        """Reference groups of a bp class from the packed library: a list of (is_purine1, is_purine2, n1, n2, coords1 (B,n1,3), coords2 (B,n2,3), names (B,)).
        References within a group share base types and atom counts so they can be superimposed in one batch. None if there is no packed library, [] if the class has no references."""
        if self._pack is None:
            self._pack = np.load(self.pack_path) if os.path.isfile(self.pack_path) else False
        if self._pack is False:
            return None
        if bp_class not in self._packed_classes:
            groups = []
            if f"{bp_class}/names" in self._pack.files:
                names, purine, counts, xyz = (self._pack[f"{bp_class}/{k}"] for k in ("names", "purine", "counts", "xyz"))
                shapes = {}
                for i in range(len(names)):
                    shapes.setdefault((bool(purine[i, 0]), bool(purine[i, 1]), int(counts[i, 0]), int(counts[i, 1])), []).append(i)
                for (p1, p2, n1, n2), idx in shapes.items():
                    groups.append((p1, p2, n1, n2, xyz[idx, 0, :n1], xyz[idx, 1, :n2], names[idx]))
            self._packed_classes[bp_class] = groups
        return self._packed_classes[bp_class]

    @staticmethod
    def batched_basepair_rmsd(cur1, cur2, r1, r2):
        """calculate_basepair_rmsd_coords for a batch of references: cur1 (B,n1,3) and cur2 (B,n2,3) against one observed pair r1 (n1,3), r2 (n2,3). Returns (B,) RMSDs."""
        def aligned_rmsd(fixed, moving, other_fixed, other_moving):
            fc = fixed.mean(axis=1, keepdims=True)
            mc = moving.mean(axis=0)
            H = np.einsum('ni,bnj->bij', moving - mc, fixed - fc)
            U, _, Vt = np.linalg.svd(H)
            d = np.sign(np.linalg.det(np.einsum('bji,bkj->bik', Vt, U)))
            D = np.zeros_like(H)
            D[:, 0, 0] = D[:, 1, 1] = 1.0
            D[:, 2, 2] = d
            R = np.einsum('bij,bjk,bkl->bil', U, D, Vt)
            t = fc - np.einsum('i,bij->bj', mc, R)[:, None, :]
            diff = other_fixed - (np.einsum('ni,bij->bnj', other_moving, R) + t)
            return np.sqrt(np.sum(diff * diff, axis=(1, 2)) / other_fixed.shape[1])
        return (aligned_rmsd(cur1, r1, cur2, r2) + aligned_rmsd(cur2, r2, cur1, r1)) / 2

    def get_rmsds_packed(self, groups, r1_type, r2_type, r1_coords, r2_coords):
        """(rmsd, curated name) for every usable reference of a class, best of direct and swapped orientation"""
        out = []
        for p1, p2, n1, n2, c1, c2, names in groups:
            best = np.full(len(names), np.inf)
            if (r1_type == "purine") == p1 and (r2_type == "purine") == p2 and len(r1_coords) == n1 and len(r2_coords) == n2:
                best = self.batched_basepair_rmsd(c1, c2, r1_coords, r2_coords)
            if (r1_type == "purine") == p2 and (r2_type == "purine") == p1 and len(r2_coords) == n1 and len(r1_coords) == n2:
                best = np.minimum(best, self.batched_basepair_rmsd(c1, c2, r2_coords, r1_coords))
            out.extend((float(r), str(n)) for r, n in zip(best, names) if np.isfinite(r))
        return out

    def get_values(self, pair: PairInformation) -> Sequence[Optional[float]]:
        bp_type = pair.type
        bp_class = f"{bp_type.type}_{resname_map.get(pair.res1.resname, pair.res1.resname)}-{resname_map.get(pair.res2.resname, pair.res2.resname)}_{bp_type.subclass}"

        packed = self.get_packed_class(bp_class)
        curated_files = self.get_class_files(bp_class) if packed is None else packed

        if not curated_files:
            return [None, None] + ([None, None] if self.use_knn_scoring else [])

        # Get atoms from the observed basepair
        r1_atoms, r1_type = self.get_specific_atoms(pair.res1)
        r2_atoms, r2_type = self.get_specific_atoms(pair.res2)

        if not r1_atoms or not r2_atoms:
            return [None, None] + ([None, None] if self.use_knn_scoring else [])

        r1_coords = self.atoms_to_coords(r1_atoms)
        r2_coords = self.atoms_to_coords(r2_atoms)

        # Calculate RMSD against ALL curated basepairs
        rmsd_results = []  # List of (rmsd, curated_file)

        if packed is not None:
            rmsd_results = self.get_rmsds_packed(packed, r1_type, r2_type, r1_coords, r2_coords)
            curated_files = []  # the per-file path below is for the unpacked fallback

        for curated_file, curated_path in curated_files:
            ref = self.get_reference(curated_path)
            if ref is None:
                continue
            cur1_type, cur2_type, cur1_coords, cur2_coords = ref

            # Try both orientations (direct and swapped); take the better (lower) RMSD
            try:
                best_rmsd = None
                if r1_type == cur1_type and r2_type == cur2_type:
                    # Direct mapping: r1->cur1, r2->cur2
                    best_rmsd = self.calculate_basepair_rmsd_coords(cur1_coords, cur2_coords, r1_coords, r2_coords)

                if r1_type == cur2_type and r2_type == cur1_type:
                    # Swapped mapping: r1->cur2, r2->cur1
                    rmsd_swapped = self.calculate_basepair_rmsd_coords(cur1_coords, cur2_coords, r2_coords, r1_coords)
                    best_rmsd = rmsd_swapped if best_rmsd is None else min(best_rmsd, rmsd_swapped)
            except Exception as e:
                print(f"Error processing curated {curated_path}: {e}")
                continue

            if best_rmsd is not None and best_rmsd != float('inf'):
                rmsd_results.append((best_rmsd, curated_file[:-4]))

        if not rmsd_results:
            return [None, None] + ([None, None] if self.use_knn_scoring else [])
        
        # Sort ALL results by RMSD (lowest first)
        rmsd_results.sort(key=lambda x: x[0])
        
        # Get the best single match
        lowest_rmsd, best_curated = rmsd_results[0]
        
        # Initialize KNN results
        knn_metric = None
        napasco = None
        
        # Calculate KNN metric if requested and we have enough results
        if self.use_knn_scoring and len(rmsd_results) >= self.k_neighbors:
            # Get the k lowest RMSDs
            k_lowest_rmsds = [rmsd for rmsd, _ in rmsd_results[:self.k_neighbors]]
            
            # Calculate KNN metric using the existing function
            knn_metric = self.calculate_sqrt_sum_reciprocal_squared(k_lowest_rmsds)
            
            # Get CDF score if KNN metric was calculated
            if knn_metric is not None:
                napasco = self.get_cdf_score(bp_class, knn_metric)
        
        # Return results
        results = [
            lowest_rmsd if lowest_rmsd != float('inf') else None,
            best_curated 
        ]

        if self.use_knn_scoring:
            results.extend([napasco, knn_metric])
                
        return results
    
@functools.lru_cache(maxsize=None)
def get_curated_metric(cdf_models_dir: str = DEFAULT_CDF_DIR, curated_dir: str = DEFAULT_CURATED_DIR) -> "CuratedRMSDMetric":
    """Returns a shared CuratedRMSDMetric, so the CDF pickles and curated structures are loaded once per process, not once per structure"""
    return CuratedRMSDMetric(cdf_models_dir, use_knn_scoring=True, curated_dir=curated_dir)

class StandardMetrics:
    Coplanarity = CoplanarityEdgeMetrics()
    Isostericity = IsostericityCoreMetrics()
    EulerClassic = EulerAngleMetrics(["euler_phi", "euler_theta", "euler_psi"], get_C1_N_euler_angles, 1)
    YawPitchRoll = EulerAngleMetrics(["yaw1", "pitch1", "roll1"], get_C1_N_yaw_pitch_roll, 1)
    YawPitchRoll2 = EulerAngleMetrics(["yaw2", "pitch2", "roll2"], get_C1_N_yaw_pitch_roll, 2)
    Translation1 = TranslationMetrics(1)
    Translation2 = TranslationMetrics(2)

def get_angle(atom1: gemmi.Atom, atom2: gemmi.Atom, atom3: gemmi.Atom) -> Optional[float]:
    """
    Calculates the 3-point angle (in degrees) using Gemmi atom positions.
    Angle is at atom2.
    """
    if atom1 is None or atom2 is None or atom3 is None:
        return None
    
    # Extract coordinates as numpy arrays
    p1 = np.array([atom1.pos.x, atom1.pos.y, atom1.pos.z])
    p2 = np.array([atom2.pos.x, atom2.pos.y, atom2.pos.z])
    p3 = np.array([atom3.pos.x, atom3.pos.y, atom3.pos.z])
    
    # Create vectors relative to the vertex (atom2)
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate dot product and magnitudes
    dot_prod = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0
        
    # Clip for numerical stability (ensures value is within [-1, 1])
    cosine_angle = np.clip(dot_prod / (mag1 * mag2), -1.0, 1.0)
    
    return math.degrees(np.arccos(cosine_angle))

def get_distance(atom1: gemmi.Atom, atom2: gemmi.Atom) -> float:
    """
    Calculates the Euclidean distance between two Gemmi atoms.
    """
    # Gemmi.Position has a built-in dist method
    return atom1.pos.dist(atom2.pos)

@dataclass
class HBondStats:
    length: Optional[float]
    donor_angle: Optional[float]
    acceptor_angle: Optional[float]
    # "out-of-plane angle" - angle between the line of the bond and the first base plane
    OOPA1: Optional[float] = None
    OOPA2: Optional[float] = None

def hbond_stats(
    atom0: Optional[gemmi.Atom], atom1: gemmi.Atom, 
    atom2: gemmi.Atom, atom3: Optional[gemmi.Atom],
    residue1_plane: Optional[ResiduePosition],
    residue2_plane: Optional[ResiduePosition]
) -> HBondStats:
    """Gets stats for a given hydrogen bond in a given basepair using Gemmi."""
    
    if atom1 is None or atom2 is None:
        return HBondStats(length=None, donor_angle=None, acceptor_angle=None)

    # Bonding distance checks (Gemmi uses .dist() for efficient Euclidean distance)
    if atom0 is not None:
        d01 = atom0.pos.dist(atom1.pos)
        assert d01 <= 1.8, f"atoms 0,1 not bonded: {atom0.name} {atom1.name} ({d01} > 1.8)"
        
    if atom3 is not None:
        d23 = atom2.pos.dist(atom3.pos)
        assert d23 <= 1.8, f"atoms 2,3 not bonded: {atom2.name} {atom3.name} ({d23} > 1.8)"

    dist = atom1.pos.dist(atom2.pos)
    if dist >= 20:
        # atoms too far apart to form an H-bond
        return HBondStats(length=None, donor_angle=None, acceptor_angle=None)

    # Calculate angles using the get_angle function rewritten for Gemmi previously
    result = HBondStats(
        length=dist,
        donor_angle=get_angle(atom0, atom1, atom2),
        acceptor_angle=get_angle(atom3, atom2, atom1)
    )

    # Out-of-Plane Angle (OOPA) calculations
    # Convert to numpy arrays for the relative_line_rotation method
    if residue1_plane is not None:
        coord1 = np.array([atom1.pos.x, atom1.pos.y, atom1.pos.z])
        coord2 = np.array([atom2.pos.x, atom2.pos.y, atom2.pos.z])
        result.OOPA1 = residue1_plane.get_relative_line_rotation(coord1, coord2)
        
    if residue2_plane is not None:
        coord1 = np.array([atom1.pos.x, atom1.pos.y, atom1.pos.z])
        coord2 = np.array([atom2.pos.x, atom2.pos.y, atom2.pos.z])
        result.OOPA2 = residue2_plane.get_relative_line_rotation(coord2, coord1)

    return result

resname_map = pair_defs.resname_map
def get_hb_atom(n, r1: AltResidue, r2: AltResidue):
    if n[0] == 'A':
        return r1.get_atom(pair_defs.map_atom_name(r1.resname, n[1:]), None)
    elif n[0] == 'B':
        return r2.get_atom(pair_defs.map_atom_name(r2.resname, n[1:]), None)
    else:
        raise ValueError(f"Invalid atom name: {n}")

def pair_hbonds_stats(pair: PairInformation) -> Optional[List[Optional[HBondStats]]]:
    """Returns stats for all H-bonds in the given basepair (assuming the `pair.type`)"""
    r1, r2 = pair.res1, pair.res2
    rp1, rp2 = pair.position1, pair.position2
    # if r1.resname > r2.resname:
    #     r1, r2 = r2, r1
    hbonds = pdef.get_hbonds(pair.type, throw=False)
    if len(hbonds) == 0:
        return None
    # print(pair_type, pair_name, hbonds)

    bonds_atoms = [
        tuple(get_hb_atom(a, r1, r2) for a in atoms)
        for atoms in hbonds
    ]

    return [
        hbond_stats(atoms[0], atoms[1], atoms[2], atoms[3], rp1, rp2)
        for atoms, hb in zip(bonds_atoms, hbonds)
    ]

def get_atom_df(model: gemmi.Model) -> dict[str, np.ndarray]:
    # In Gemmi, we can count atoms efficiently
    atom_count = model.count_atoms()
    
    # Pre-allocate arrays using the new naming schema
    # S4/S8 used for fixed-length byte strings to keep memory low
    auth_seq_id = np.zeros(atom_count, dtype=np.int32)
    label_seq_id = np.zeros(atom_count, dtype=np.int32)
    resname = np.zeros(atom_count, dtype="S4")
    auth_asym_id = np.zeros(atom_count, dtype="S8")
    label_asym_id = np.zeros(atom_count, dtype="S8")
    coord = np.zeros((atom_count, 3), dtype=np.float32)
    element = np.zeros(atom_count, dtype="S2")
    atom_name = np.zeros(atom_count, dtype="S4")

    atom_idx = 0
    for chain in model:
        # auth_asym_id is usually chain.name
        # label_asym_id is usually retrieved from the mmCIF data, 
        # but in Gemmi, chain.name is the standard identifier.
        c_name = chain.name.encode('utf-8')
        
        for residue in chain:
            # auth_seq_id = residue.seqid.num
            # label_seq_id = residue.label_seq (if available, else falls back to num)
            r_auth = residue.seqid.num
            r_label = residue.label_seq if residue.label_seq is not None else r_auth
            r_name = residue.name.encode('utf-8')
            
            # auth_asym_id vs label_asym_id logic
            # Using residue.subchain as the "auth" identifier if it exists
            r_auth_chain = (residue.subchain if residue.subchain else chain.name).encode('utf-8')

            for atom in residue:
                auth_seq_id[atom_idx] = r_auth
                label_seq_id[atom_idx] = r_label
                resname[atom_idx] = r_name
                auth_asym_id[atom_idx] = r_auth_chain
                label_asym_id[atom_idx] = c_name
                
                # Gemmi positions
                coord[atom_idx, 0] = atom.pos.x
                coord[atom_idx, 1] = atom.pos.y
                coord[atom_idx, 2] = atom.pos.z
                
                element[atom_idx] = atom.element.name.encode('utf-8')
                atom_name[atom_idx] = atom.name.encode('utf-8')

                atom_idx += 1

    return {
        "auth_asym_id": auth_asym_id,
        "label_asym_id": label_asym_id,
        "auth_seq_id": auth_seq_id,
        "label_seq_id": label_seq_id,
        "resname": resname,
        "coord": coord,
        "element": element,
        "name": atom_name,
    }

def to_csv_row(df: pl.DataFrame, i: int = 0, max_len = None) -> str:
    row = next(df[i, :].iter_rows())
    if max_len is not None:
        row = row[:max_len]
    return ','.join(str(x) for x in row)

T = TypeVar("T")
def lazy(create: Callable[..., T], *args) -> Callable[[], T]:
    """
    Lazily creates a value using the given function and arguments

    val = lazy(lambda: create_something())
    ...
    val().do_something() # create_something() is called here
    val().do_something() # create_something() is not called again
    """
    cache: T
    has_value = False
    def core():
        nonlocal cache, has_value
        if not has_value:
            cache = create(*args)
            has_value = True
        return cache
    return core

def get_residue(structure: gemmi.Structure, 
                strdata: Optional[Callable[[], pdb_utils.StructureData]], 
                model_idx: int, 
                auth_chain: str, 
                resname: str, 
                inter_nr: Any, 
                nr: int, 
                ins: Optional[str], 
                alt: Optional[str], 
                symop: Optional[str]) -> Optional[AltResidue]:

    try:
        model = structure[model_idx - 1]
    except IndexError:
        raise KeyError(f"Model index {model_idx} out of range.")

    try:
        ch = model[auth_chain]
    except KeyError:
        available_chains = [c.name for c in model]
        raise KeyError(f"Chain {auth_chain} not found. Available: {available_chains}")

    target_nr = int(nr)
    target_ins = ins if ins and ins.strip() else ' '
    
    found = None
    for r in ch:
        if r.seqid.num == target_nr and r.seqid.icode == target_ins:
            found = r
            break

    if found is None:
        raise KeyError(f"Residue {auth_chain}.{nr}{target_ins} not found")
    
    if symop and symop != '1_555':
        if strdata is not None:
            sym_transform = next((a for a in strdata().assembly 
                                  if a.pdbname == symop or f"ASM_{a.id}" == symop), None)
            
            if sym_transform is not None:
                found = found.clone()
                # Crystallographic symop: new_pos = R @ pos + t
                # _apply_transformation does: rot @ (pos + t1) + t2, so t1=0, t2=translation
                _apply_transformation(found, np.zeros(3),
                                      sym_transform.rotation, sym_transform.translation)
            else:
                return None
            
    return AltResidue(found, alt or '')

def get_stats_for_csv(df_: pl.DataFrame, 
                      structure: Any, 
                      strdata: Callable[[], Any], 
                      metrics: list, 
                      max_hb_length: Optional[float]) -> Iterator[tuple]:
    
    df = df_.with_row_index("row_nr")
    family_col = df["family" if "family" in df.columns else "type"]
    
    def residue_cache_create_entry(model, auth_chain, res, nr, ins, alt, symop):
        try:
            # alt_res is an AltResidue object (contains .res and .alt)
            alt_res = get_residue(structure, strdata, model, auth_chain, res, None, nr, ins, alt, symop)
            if alt_res is None:
                return None
            
            info = ResidueInformation(alt_res)
            r = alt_res.res # This is the Gemmi residue
            
            # Use .alt instead of .altloc based on your error log
            meta = {
                "label_alt_id": alt_res.alt, 
                "label_comp_id": r.name,
                "label_asym_id": r.subchain,
                "label_entity_id": r.entity_id,
                "label_seq_id": r.label_seq if r.label_seq != -1 else None,
                "pdbx_PDB_ins_code": r.seqid.icode.strip() if r.seqid.icode != ' ' else '',
                "auth_seq_id": r.seqid.num,
                "auth_comp_id": r.name,
                "auth_asym_id": auth_chain 
            }
            return (info, meta)
        except Exception as e:
            print(f"ERROR in residue_cache_create_entry: model={model}, chain={auth_chain}, nr={nr}, ins={ins}, alt={repr(alt)}", file=sys.stderr)
            print(f"  Exception: {type(e).__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return None

    # Identify unique residues to cache
    residues_df = pl.concat([
        df.select(m=pl.col("model"), ch=pl.col("auth_asym_id1"), r=pl.col("auth_comp_id1"), nr=pl.col("auth_seq_id1"), ins=pl.col("pdbx_PDB_ins_code1"), alt=pl.col("label_alt_id1"), sym=pl.col("symmetry_operation1")),
        df.select(m=pl.col("model"), ch=pl.col("auth_asym_id2"), r=pl.col("auth_comp_id2"), nr=pl.col("auth_seq_id2"), ins=pl.col("pdbx_PDB_ins_code2"), alt=pl.col("label_alt_id2"), sym=pl.col("symmetry_operation2"))
    ]).unique()

    residue_cache = {
        (row['m'], row['ch'], row['r'], row['nr'], (row['ins'] or '').strip(), (row['alt'] or '').strip(), row['sym']): 
        residue_cache_create_entry(row['m'], row['ch'], row['r'], row['nr'], row['ins'], row['alt'], row['sym'])
        for row in residues_df.iter_rows(named=True)
    }

    for i, row in enumerate(df.iter_rows(named=True)):
        try:
            k1 = (row["model"], row["auth_asym_id1"], row["auth_comp_id1"], row["auth_seq_id1"], (row["pdbx_PDB_ins_code1"] or '').strip(), (row["label_alt_id1"] or '').strip(), row["symmetry_operation1"])
            k2 = (row["model"], row["auth_asym_id2"], row["auth_comp_id2"], row["auth_seq_id2"], (row["pdbx_PDB_ins_code2"] or '').strip(), (row["label_alt_id2"] or '').strip(), row["symmetry_operation2"])
            
            res1_data, res2_data = residue_cache.get(k1), residue_cache.get(k2)
            if not res1_data or not res2_data: continue

            r1_info, m1 = res1_data
            r2_info, m2 = res2_data

            pair_type = pair_defs.PairType.create(family_col[i], row["auth_comp_id1"], row["auth_comp_id2"], row["subclass"], name_map=resname_map)
            pair_info = PairInformation(pair_type, r1_info, r2_info)
            
            hbonds = pair_hbonds_stats(pair_info)
            if hbonds is None: continue
            
            metric_values = []
            for m in metrics:
                metric_values.extend(m.get_values(pair_info))

            l_meta = {**{f"{k}1": v for k, v in m1.items()}, **{f"{k}2": v for k, v in m2.items()}}
            yield row["row_nr"], hbonds, metric_values, l_meta
        except Exception as e:
            print(f"Error processing pair at row {i}: {type(e).__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            continue

def remove_duplicate_pairs_phase1(df: pl.DataFrame):
    """
    Removes duplicate symmetric pairs using updated Gemmi column names.
    Keeps preferred family (W > H > S) and base (G > A > C > U) ordering.
    """
    family_col_name = "family" if "family" in df.columns else "type"
    family_col = pl.col(family_col_name)
    original_len = len(df)
    
    # Updated unique check with new column names
    id_cols = [
        family_col_name, "subclass", "pdbid", "model", 
        "auth_asym_id1", "auth_comp_id1", "auth_seq_id1", "pdbx_PDB_ins_code1", "label_alt_id1",
        "auth_asym_id2", "auth_comp_id2", "auth_seq_id2", "pdbx_PDB_ins_code2", "label_alt_id2",
        "symmetry_operation1", "symmetry_operation2", "symmetry_id1", "symmetry_id2"
    ]
    
    df = df.unique(id_cols)
    if len(df) < original_len:
        print(f"Removed duplicates - unique {original_len} -> {len(df)}")
        original_len = len(df)
        
    df = df.with_row_index("_tmp_row_nr")
    
    # Reverse the Leontis-Westhof notation (e.g., cWS -> cSW)
    df = df.with_columns(
        family_col.str.replace("^(n?[ct])([WHSBwhsb])([WHSBwhsb])([a-z]*)$", "$1$3$2$4").alias("_tmp_family_reversed")
    )

    def core(df: pl.DataFrame, condition):
        # Join the DF against itself with reversed identifiers to find symmetric pairs
        duplicated = df.filter(condition).join(
            df,
            left_on=[
                family_col_name, 'subclass', 'pdbid', 'model', 
                'auth_asym_id1', 'auth_comp_id1', 'auth_seq_id1', 'pdbx_PDB_ins_code1', 'label_alt_id1', 
                'auth_asym_id2', 'auth_seq_id2', 'pdbx_PDB_ins_code2', 'label_alt_id2'
            ],
            right_on=[
                '_tmp_family_reversed', 'subclass', 'pdbid', 'model', 
                'auth_asym_id2', 'auth_comp_id2', 'auth_seq_id2', 'pdbx_PDB_ins_code2', 'label_alt_id2', 
                'auth_asym_id1', 'auth_seq_id1', 'pdbx_PDB_ins_code1', 'label_alt_id1'
            ],
            join_nulls=True, how="inner"
        )
        
        duplicated_f = duplicated.select("_tmp_row_nr", "_tmp_row_nr_right")
        duplicated_set = set(duplicated_f["_tmp_row_nr"])
        check_not_removed = set(duplicated_f["_tmp_row_nr_right"])
        
        if duplicated_set.intersection(check_not_removed):
            duplicated_set.difference_update(check_not_removed)

        return df.filter(pl.col("_tmp_row_nr").is_in(duplicated_set).not_())

    # Priority 1: Family orientation (prefer WS over SW)
    df = core(df, family_col.str.contains("^(?i)n?(c|t)(SH|HW|SW|BW)a?$"))

    # Priority 2: Case preference (prefer uppercase/canonical over lowercase)
    df = core(df, family_col.str.contains("^n?[ct][whsb][WHSB]a?$"))

    # Priority 3: Base ordering (G > A > C > U)
    base_ordering = {'A': 1, 'G': 2, 'C': 3, 'U': 4, 'T': 4}
    df = core(df,
        family_col.str.to_lowercase().is_in(["tss", "css"]).not_() &
        (pl.col("auth_comp_id1").replace(resname_map).replace(base_ordering, default=0) > 
         pl.col("auth_comp_id2").replace(resname_map).replace(base_ordering, default=0))
    )

    return df.drop(["_tmp_row_nr", "_tmp_family_reversed"])

def remove_duplicate_pairs(df: pl.DataFrame):
    """
    Final deduplication. If ordering is still ambiguous, 
    keeps the pair with shorter H-bonds or lower residue number.
    """
    df = remove_duplicate_pairs_phase1(df)
    family_col_name = "family" if "family" in df.columns else "type"

    # Create a unique ID for the pair regardless of sequence order
    def create_pair_id_expr():
        # This expression creates a string ID where the smaller residue ID always comes first.
        # fill_null("") is required because pdbx_PDB_ins_code is null for most residues,
        # and pl.format returns null when any argument is null, which would collapse all
        # altloc variants into one row in the subsequent unique() call.
        res1_id = pl.format("{}:{}{}{}", "auth_asym_id1", "auth_seq_id1",
                            pl.col("pdbx_PDB_ins_code1").fill_null(""),
                            pl.col("label_alt_id1").fill_null(""))
        res2_id = pl.format("{}:{}{}{}", "auth_asym_id2", "auth_seq_id2",
                            pl.col("pdbx_PDB_ins_code2").fill_null(""),
                            pl.col("label_alt_id2").fill_null(""))
        
        return pl.concat_str([
            family_col_name,
            pl.lit("_"),
            pl.when(res1_id < res2_id).then(res1_id + "|" + res2_id).otherwise(res2_id + "|" + res1_id)
        ]).alias("_tmp_pair_id")

    df = df.with_columns(create_pair_id_expr()).with_row_index("_tmp_row_nr")

    # Scoring: shorter H-bonds are better (lower score)
    score = pl.lit(0, dtype=pl.Float64)
    null_count = pl.lit(0, dtype=pl.Int64)
    
    for col in df.columns:
        if col.startswith("hb_") and col.endswith("_length"):
            score += pl.col(col).fill_nan(4.0).fill_null(4.0)
            null_count += (pl.col(col).is_null() | pl.col(col).is_nan()).cast(pl.Int64)
        if col.startswith("dssr_"):
            score += pl.col(col).is_null().cast(pl.Float64) * 0.03

    # Symmetric definitions (e.g., cWW A-A) don't need distance scoring for orientation
    sym_pt = {x.lower() for x in pair_defs.defined_pair_types() if pair_defs.has_symmetrical_definition(x)}
    pair_str = (pl.col(family_col_name) + "_" + pl.col("auth_comp_id1").replace(resname_map) + "-" + pl.col("auth_comp_id2").replace(resname_map)).str.to_lowercase()
    
    score = pl.when(pair_str.is_in(sym_pt)).then(pl.lit(0.0)).otherwise(score)

    # Sort and keep the "best" representative
    # base_pref: prefer A(1) > G(2) > C(3) > U(4) as residue 1 when scores are equal
    base_ordering_map = {'A': 1, 'DA': 1, 'G': 2, 'DG': 2, 'C': 3, 'DC': 3, 'U': 4, 'DU': 4, 'T': 4, 'DT': 4}
    base_pref = pl.col("auth_comp_id1").replace(resname_map).replace(base_ordering_map, default=0)
    df = df.sort([null_count, score, "subclass", base_pref, "auth_asym_id1", "auth_seq_id1", "pdbx_PDB_ins_code1", "label_alt_id1"])
    df = df.unique([family_col_name, "subclass", "pdbid", "model", "_tmp_pair_id"], keep="first")
    
    return df.sort("_tmp_row_nr").drop(["_tmp_pair_id", "_tmp_row_nr"])

def postfilter_hb(df: pl.DataFrame, length: Optional[float]):
    if length:
        df = df.filter(pl.any_horizontal(pl.col("^hb_\\d+_length$") <= length))
    return df

def postfilter_shift(df: pl.DataFrame, shift: Optional[float]):
    if shift:
        df = df.filter((pl.col("coplanarity_shift1").abs() < shift) & (pl.col("coplanarity_shift2").abs() < shift))
    return df

def get_max_bond_count(df: pl.DataFrame):
    """Gets the maximum number of hydrogen bonds in the given DataFrame.  """
    return max(len(x) for x in pair_defs.hbonding_atoms.values())

def make_backbone_columns(df: pl.DataFrame, structure: Optional[gemmi.Structure]) -> list[pl.Series]:
    """Calculates is_dinucleotide and is_parallel using Gemmi indexing."""
    is_dinucleotide = [False] * len(df)
    is_parallel: list[Optional[bool]] = [None] * len(df)

    if structure is not None:
        # Create a mapping: (model_idx, chain_name) -> {(auth_seq, ins): linear_index}
        # This linear_index is the position in the chain (ignoring water/HOH)
        chain_index = {}
        for m_idx, model in enumerate(structure, 1):
            for chain in model:
                res_map = {}
                # Filter out waters for clean backbone adjacency
                filtered_residues = [r for r in chain if r.name != 'HOH']
                for i, res in enumerate(filtered_residues):
                    res_map[(res.seqid.num, res.seqid.icode.strip() or ' ')] = i
                chain_index[(m_idx, chain.name)] = res_map

        # Create a spatial lookup of the pairs currently in the dataframe
        # Key: (model, chain1, res_idx1, chain2, res_idx2)
        pair_set = set()
        for row in df.iter_rows(named=True):
            m, c1, c2 = row["model"], row["auth_asym_id1"], row["auth_asym_id2"]
            n1, i1 = row["auth_seq_id1"], (row["pdbx_PDB_ins_code1"] or ' ').strip() or ' '
            n2, i2 = row["auth_seq_id2"], (row["pdbx_PDB_ins_code2"] or ' ').strip() or ' '

            idx1 = chain_index.get((m, c1), {}).get((n1, i1))
            idx2 = chain_index.get((m, c2), {}).get((n2, i2))
            
            if idx1 is not None and idx2 is not None:
                pair_set.add((m, c1, idx1, c2, idx2))

        # Determine relationships
        for i, row in enumerate(df.iter_rows(named=True)):
            m, c1, c2 = row["model"], row["auth_asym_id1"], row["auth_asym_id2"]
            idx1 = chain_index.get((m, c1), {}).get((row["auth_seq_id1"], (row["pdbx_PDB_ins_code1"] or ' ').strip() or ' '))
            idx2 = chain_index.get((m, c2), {}).get((row["auth_seq_id2"], (row["pdbx_PDB_ins_code2"] or ' ').strip() or ' '))

            if idx1 is None or idx2 is None:
                continue

            # Check if residues are neighbors on the same chain
            if c1 == c2:
                is_dinucleotide[i] = abs(idx2 - idx1) == 1

            # Parallel/Antiparallel logic
            # Parallel: (i+1, j+1) exists or (i-1, j-1) exists
            par = (m, c1, idx1 + 1, c2, idx2 + 1) in pair_set or (m, c1, idx1 - 1, c2, idx2 - 1) in pair_set
            # Antiparallel: (i+1, j-1) exists or (i-1, j+1) exists
            antipar = (m, c1, idx1 + 1, c2, idx2 - 1) in pair_set or (m, c1, idx1 - 1, c2, idx2 + 1) in pair_set

            if antipar and not par:
                is_parallel[i] = False
            elif par and not antipar:
                is_parallel[i] = True

    return [
        pl.Series("is_dinucleotide", is_dinucleotide, dtype=pl.Boolean),
        pl.Series("is_parallel", is_parallel, dtype=pl.Boolean)
    ]

def make_stats_columns(pdbid: str, df: pl.DataFrame, add_metadata_columns: bool, 
                       max_bond_count: int, metrics: list, 
                       max_hb_length: Optional[float] = None, 
                       structure: Optional[Any] = None) -> Tuple[str, pl.DataFrame, np.ndarray]:
    
    bond_params = [x.name for x in dataclasses.fields(HBondStats)]
    valid = np.zeros(len(df), dtype=np.bool_)
    
    hb_columns = [(f"hb_{i}_{p}", np.full(len(df), np.nan, dtype=np.float32)) 
                  for i in range(max_bond_count) for p in bond_params]

    metric_names = [name for m in metrics for name in m.get_columns()]
    _string_metric_names = {'curated_file'}
    metric_columns = {
        name: ([None] * len(df) if name in _string_metric_names else np.full(len(df), np.nan, dtype=np.float32))
        for name in metric_names
    }

    # All 18 IDs (9 per residue, as auth_atom_id was removed)
    id_reporting_cols = {
        "label_alt_id1": [None] * len(df), "label_comp_id1": [None] * len(df), "label_asym_id1": [None] * len(df),
        "label_entity_id1": [None] * len(df), "label_seq_id1": np.zeros(len(df), dtype=np.int32),
        "pdbx_PDB_ins_code1": [None] * len(df), "auth_seq_id1": np.zeros(len(df), dtype=np.int32),
        "auth_comp_id1": [None] * len(df), "auth_asym_id1": [None] * len(df),
        "label_alt_id2": [None] * len(df), "label_comp_id2": [None] * len(df), "label_asym_id2": [None] * len(df),
        "label_entity_id2": [None] * len(df), "label_seq_id2": np.zeros(len(df), dtype=np.int32),
        "pdbx_PDB_ins_code2": [None] * len(df), "auth_seq_id2": np.zeros(len(df), dtype=np.int32),
        "auth_comp_id2": [None] * len(df), "auth_asym_id2": [None] * len(df)
    }

    structure_data = lambda: pdb_utils.load_sym_data(None, pdbid)
    if structure is None:
        structure = pdb_utils.load_pdb(None, pdbid)

    if structure is not None:
        for i, hbonds, metric_values, l_meta in get_stats_for_csv(df, structure, structure_data, metrics, max_hb_length):
            valid[i] = any(hb and hb.length and hb.length <= 4.5 for hb in hbonds)

            for key in id_reporting_cols:
                if key in l_meta:
                    id_reporting_cols[key][i] = l_meta[key]

            for j, m_val in enumerate(metric_values):
                metric_columns[metric_names[j]][i] = m_val

            for j, s in enumerate(hbonds[:max_bond_count]):
                if s:
                    for p_idx, p_name in enumerate(bond_params):
                        hb_columns[j * len(bond_params) + p_idx][1][i] = getattr(s, p_name)

    # Convert to Polars Series
    series_list = []
    for name, data in id_reporting_cols.items():
        series_list.append(pl.Series(name, data, dtype=pl.Int32 if "seq_id" in name else pl.Utf8))
    for name, array in hb_columns:
        series_list.append(pl.Series(name, array, dtype=pl.Float32))
    for name, array in metric_columns.items():
        if isinstance(array, list):
            series_list.append(pl.Series(name, array, dtype=pl.Utf8))
        else:
            series_list.append(pl.Series(name, array, dtype=pl.Float32))

    stats_df = pl.DataFrame(series_list)
    
    # Merge with original data, prioritizing existing metadata
    cols_to_add = [c for c in stats_df.columns if c not in df.columns]
    result_df = pl.concat([df, stats_df.select(cols_to_add)], how="horizontal")
    
    return pdbid, result_df, valid

def df_hstack(columns: list[pl.DataFrame]) -> pl.DataFrame:
    return functools.reduce(pl.DataFrame.hstack, columns)

def load_input_pdbids(inputs: list[str]) -> list[str]:
    r = set()
    i: str
    for i in inputs:
        if i.endswith(".parquet"):
            r.update(pl.scan_parquet(i, cache=False, low_memory=True).select(pl.col("pdbid").str.to_lowercase()).unique().collect(streaming=True)["pdbid"].to_list())
        elif i.endswith(".csv"):
            r.update(pl.scan_parquet(i, cache=False, low_memory=True).select(pl.col("pdbid").str.to_lowercase()).unique().collect(streaming=True)["pdbid"].to_list())
        elif i.endswith("_basepair.txt") or i.endswith("_basepair_detail.txt"):
            r.add(os.path.basename(i).split("_")[0].lower())
        elif (m := re.search(r"(\w{4})[.](pdb|cif|mmcif)([.](gz|zstd?))?$", i, re.IGNORECASE)):
            r.add(m.group(1))
        elif (m := re.match(r"\w{4}", i)):
            r.add(i)
        else:
            raise ValueError(f"Unknown input file type: {i}")
    return list(sorted(r))

def override_pair_family(df: pl.DataFrame, fam: Optional[str], boundaries: pl.DataFrame) -> pl.DataFrame:
    if not fam:
        return df
    
    # Drop columns if they exist
    cols_to_drop = [col for col in ["type", "family"] if col in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop)
    
    override: list[str] = fam.split(",")
    
    if len(override) == 1:
        # Get subclasses for this single family from boundaries
        subclasses = boundaries.filter(pl.col('family') == override[0]).select('subclass').unique()['subclass'].to_list()
        families_dict = [{"type": override[0], "family": override[0], "subclass": str(sc)} for sc in subclasses]
        
        x = len(df)
        df = df.join(pl.DataFrame(families_dict), how='cross')
        assert x * len(families_dict) == len(df)
    else:
        # Get all family-subclass combinations from boundaries
        families_data = boundaries.filter(pl.col('family').is_in(override)).select(['family', 'subclass']).unique()
        families_dict = [
            {"type": row['family'], "family": row['family'], "subclass": str(row['subclass'])}
            for row in families_data.iter_rows(named=True)
        ]
        
        x = len(df)
        df = df.join(pl.DataFrame(families_dict), how='cross')
        assert x * len(families_dict) == len(df)
    
    return df

def load_inputs(pool: Union[Pool, Any], args, pdbid_partition_str: str = '') -> Tuple[pl.DataFrame, dict[str, Optional[str]]]:
    """
    Loads basepair identifiers from CSV/Parquet/FR3D files and prepares them 
    for Gemmi processing using mmCIF naming conventions.
    """
    inputs: List[str] = args.inputs
    if not inputs:
        raise ValueError("No input files specified")

    # 1. Define the Partition Filter
    if pdbid_partition_str:
        if '/' in pdbid_partition_str:
            # Hash-based partitioning (e.g., "0/100")
            p_ix, p_total = map(int, pdbid_partition_str.split('/'))
            if p_ix >= p_total:
                raise ValueError(f"Invalid partition: {pdbid_partition_str}")
            pdbid_filter = pl.col('pdbid').str.to_lowercase().hash(seed=418).mod(p_total) == p_ix
        else:
            # Prefix-based partitioning (e.g., "1a")
            pdbid_filter = pl.col('pdbid').str.to_lowercase().str.starts_with(pdbid_partition_str.lower())
    else:
        pdbid_filter = pl.lit(True)

    # 2. Categorize Inputs
    in_fr3d = [i for i in inputs if i.endswith("_basepair.txt") or i.endswith("_basepair_detail.txt")]
    in_pq = [i for i in inputs if i.endswith(".parquet") or i.endswith(".csv")]
    in_pdb = [i for i in inputs if re.search(r"(\w{4})[.](pdb|cif|mmcif)([.](gz|zstd?))?$", i, re.IGNORECASE)]
    in_pdbid = [i for i in inputs if re.match(r"^\w{4}$", i)]

    # Map PDBIDs to local files if provided
    pdbid_loads = {
        **{x: None for x in in_pdbid},
        **{load_input_pdbids([x])[0]: x for x in in_pdb}
    }

    results = []

    # 3. Process Table-based Inputs (CSV/Parquet)
    if in_pq:
        print(f'Scanning basepair tables with filter: {pdbid_partition_str}')
        df_lazy = scan_pair_csvs(in_pq)
        
        # Apply normalization and partition filters
        df_lazy = df_lazy.with_columns(pdbid=pl.col("pdbid").str.to_lowercase())
        df_lazy = df_lazy.filter(pdbid_filter)

        # Normalize Altlocs and Insertion Codes to be mmCIF/Gemmi friendly
        # Gemmi uses '\0' or ' ' for empty; we standardize to '' for the wrapper
        df_lazy = df_lazy.with_columns(
            label_alt_id1=pl.when(pl.col("label_alt_id1").is_in(['\0', '?', ' ']))
                           .then(pl.lit(''))
                           .otherwise(pl.col("label_alt_id1").fill_null('').str.strip_chars()),
            label_alt_id2=pl.when(pl.col("label_alt_id2").is_in(['\0', '?', ' ']))
                           .then(pl.lit(''))
                           .otherwise(pl.col("label_alt_id2").fill_null('').str.strip_chars()),
            pdbx_PDB_ins_code1=pl.col("pdbx_PDB_ins_code1").fill_null('').str.strip_chars(),
            pdbx_PDB_ins_code2=pl.col("pdbx_PDB_ins_code2").fill_null('').str.strip_chars(),
        )

        # Handle legacy "pdbsymstr" if migrating from older datasets
        if "pdbsymstr" in df_lazy.columns:
            df_lazy = df_lazy.with_columns(
                symmetry_operation1=pl.lit(None, pl.Utf8),
                symmetry_operation2=pl.when(pl.col("pdbsymstr") == "1_555")
                                     .then(pl.lit(None, pl.Utf8))
                                     .otherwise(pl.col("pdbsymstr"))
            )

        # Gemmi-friendly residue filtering (standardizing on IUPAC names)
        allowed_residues = ["A", "T", "G", "U", "C", "DA", "DT", "DG", "DC", "DU"]
        df_lazy = df_lazy.filter(
            pl.col("auth_comp_id1").is_in(allowed_residues) & 
            pl.col("auth_comp_id2").is_in(allowed_residues)
        )

        # Sort by physical location in structure for better cache locality during analysis
        df_lazy = df_lazy.sort('pdbid', 'model', 'auth_asym_id1', 'auth_seq_id1')
        
        df = df_lazy.collect(streaming=True)
        print(f"Loaded {len(df)} basepairs from tabular files.")
        results.append(df)

    else:
        raise ValueError("No valid input files (CSV, Parquet, or FR3D) identified.")

    # 5. Final Assembly
    df = results[0] if len(results) == 1 else pl.concat(results, how='diagonal_relaxed')

    # Check for mandatory columns
    if "type" not in df.columns and "family" not in df.columns:
        if not args.override_pair_family:
            raise ValueError("DataFrame missing 'family' column. Use --override-pair-family.")

    df = override_pair_family(df, args.override_pair_family)
    
    return df, pdbid_loads

def validate_missing_columns(chunks: list[pl.DataFrame]):
    """Ensures all processed chunks have consistent column schemas before concatenation."""
    all_columns = set(col for chunk in chunks for col in chunk.columns)
    for c in chunks:
        if not all_columns.issubset(c.columns):
            missing = all_columns.difference(set(c.columns))
            pdb_ids = c['pdbid'].unique().to_list()[:5]
            print(f"Chunk starting with {pdb_ids} is missing columns: {missing}")

def main_partition(pool, args, pdbid_partition='', ideal_basepairs=None):
    # Load Gemmi-identified pairs (from pair_finding.py output)
    df, bootstrap_structures = load_inputs(pool, args, pdbid_partition_str=pdbid_partition)
    
    if len(bootstrap_structures) > 0:
        raise ValueError("Direct loading of PDB files is not supported here. Run pair_finding.py first.")
    
    if args.export_only:
        return df

    raw_df_len = len(df)
    
    # Filter cross-symmetry: strictly requires at least one partner in the asymmetric unit (ASU)
    if args.disable_cross_symmetry:
        df = df.filter(pl.col("symmetry_operation1").is_null() & pl.col("symmetry_operation2").is_null())
    else:
        # Optimization: Remove cases where both residues are symmetry-generated (pure crystal contacts)
        if 'symmetry_operation1' in df.columns and 'symmetry_operation2' in df.columns:
            df = df.filter(pl.col("symmetry_operation1").is_null() | pl.col("symmetry_operation2").is_null())

    # Phase 1 Deduplication (Family and Base ordering)
    if args.dedupe:
        df = remove_duplicate_pairs_phase1(df)

    max_bond_count = get_max_bond_count(df)
    total_row_count = len(df)
    
    # Group by PDBID to process one structure at a time
    # Update sorting to use Gemmi/mmCIF column names
    groups = list(df.group_by(pl.col("pdbid")))
    max_group_size = 100_000
    
    processed_groups = []
    for pdbid, group in groups:
        # Sort for deterministic chunking
        group = group.sort('model', 'auth_asym_id1', 'auth_seq_id1', 'auth_asym_id2', 'auth_seq_id2')
        # Split massive PDBs (like ribosomes) into manageable slices
        num_slices = math.ceil(len(group) / max_group_size)
        for slice_df in group.iter_slices(len(group) // num_slices + 5):
            processed_groups.append((pdbid, slice_df))
    
    # Sort groups by size descending to prevent "tailing" in parallel execution
    processed_groups.sort(key=lambda x: -len(x[1]))
    del df

    # Define Metrics (Using the Gemmi-based StandardMetrics)
    pair_metrics = [
        StandardMetrics.Coplanarity,
        StandardMetrics.Isostericity,
        StandardMetrics.EulerClassic,
        StandardMetrics.YawPitchRoll,
        StandardMetrics.Translation1,
        StandardMetrics.Translation2,
    ]
    # Asynchronous Execution
    processes = []
    processed_row_count = {"metrics": 0}

    def report_progress(pdbid, group_size):
        processed_row_count["metrics"] += group_size
        prog = (processed_row_count["metrics"] / total_row_count) * 100
        print(f"Done: {pdbid} ({prog:.1f}%)")

    # Metrics Calculation Task
    metric_tasks = [
        pool.apply_async(make_stats_columns, 
                         args=[pdbid, group, args.metadata, max_bond_count, pair_metrics, args.postfilter_hb],
                         callback=lambda x, p=pdbid, g=len(group): report_progress(p, g))
        for (pdbid,), group in processed_groups
    ]
    processes.append(metric_tasks)

    # Post-processing helper
    def postfilter(chunk_df: pl.DataFrame):
        chunk_df = postfilter_hb(chunk_df, args.postfilter_hb)
        if args.dedupe:
            chunk_df = remove_duplicate_pairs(chunk_df)
        return chunk_df

    # Collate Results
    result_chunks = []
    for ((_pdbid,), group), task_group in zip(processed_groups, zip(*processes)):
        # task_group contains results from [metrics_task, dssr_task]
        results = [t.get() for t in task_group]
        
        # results[0] is (pdbid, added_cols_df, valid_mask)
        added_columns = [r[1] for r in results]
        valid_masks = [r[2] for r in results]

        # Combine original group data with new metrics/DSSR columns
        chunk = pl.concat([group] + added_columns, how="horizontal")
        
        if args.filter:
            # Only keep rows considered valid by all processing steps
            combined_valid = functools.reduce(np.logical_and, valid_masks)
            chunk = chunk.filter(combined_valid)
        
        chunk = postfilter(chunk)
        if len(chunk) > 0:
            result_chunks.append(chunk)

    if not result_chunks:
        return None

    validate_missing_columns(result_chunks)
    final_df = pl.concat(result_chunks)
    
    # Final cleanup and sort
    final_df = final_df.sort('pdbid', 'model', 'auth_asym_id1', 'auth_seq_id1', 'auth_asym_id2', 'auth_seq_id2')
    return final_df

def main(pool: Union[Pool, MockPool], args):
    if args.reference_basepairs:
        print("Loading metadata from ", args.reference_basepairs)
    else:
        ideal_basepairs = dict()

    if args.partition_input_select and args.partition_input:
        raise ValueError("Cannot specify both --partition-input and --partition-input-select")
    
    if args.partition_input == 0:
        select = args.partition_input_select
        if re.match(r'^\d+-\d+/\d+$', select):
            start_end, total = select.split('/')
            start, end = start_end.split('-')
            start = int(start)
            end = int(end)
            total = int(total)
            partitions = [ f"{i}/{total}" for i in range(start, end) ]
            print("Will run for multiple partitions:", partitions)
        else:
            partitions = [ select ]
        for partition in partitions:
            df = main_partition(pool, args, partition, ideal_basepairs)
            if df is not None and len(df) > 0:
                save_output(args, df.lazy(), partition)
    else:
        pdbids = load_input_pdbids(args.inputs)
        partitions = {}
        for pdbid in pdbids:
            x = pdbid[:args.partition_input]
            partitions[x] = partitions.get(x, 0) + 1
        print(f"Partitioning {len(pdbids)} PDBIDs into {len(partitions)} partitions: {partitions}")
        p_results = dict()
        try:
            for p in sorted(partitions.keys()):
                print("Processing partition", p)
                p_result = main_partition(pool, args, p, ideal_basepairs)
                if df is not None and len(p_result) > 0:
                    file = f"{args.output}_part{p}"
                    p_result.write_parquet(file)
                    p_results[p] = file

            all_df = pl.concat([ pl.scan_parquet(p, cache=False, low_memory=True) for p in p_results.values() ])
            save_output(args, all_df)
        finally:
            for p in p_results.values():
                os.remove(p)

all_familites = "cWW,cWW,tWW,tWW,cWH,tWH,cWS,tWS,cHH,cHH,tHH,cHS,tHS,cSS,tSS"

def save_output(args, df: pl.LazyFrame, partition_select = ''):
    file = args.output
    if file == "/dev/null":
        return
    if partition_select:
        x, ext = os.path.splitext(file)
        file = f"{x}_p{partition_select.replace('/', 'of')}{ext}"
    if file.endswith(".parquet"):
        df.sink_parquet(file)
    else:
        df.sink_csv(file if file.endswith(".csv") else file + ".csv")
        df.sink_parquet(file + ".parquet")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="""
        Adds geometric information to the specified basepairing CSV/Parquet/FR3D files.
        Added columns:
            * hb_0_length, hb_0_acceptor_angle, hb_0_donor_angle, ... - hydrogen bond lengths and angles between heavy atoms on both sides
            * hb_0_OOPA1, hb_0_OOPA2, ... - hydrogen bond / base plane angles
            * coplanarity_angle - angle between normals of the two base planes
            * coplanarity_shift1, coplanarity_shift2 - (minimum) distance between the pairing edge and the other base plane
            * coplanarity_edge_angle1 - angle between the pairing edge and the other base plane
            * C1_C1_distance - distance between the C1' atoms
            * C1_C1_total_angle - angle between the C1'-N bonds
            * Relative positions of N - C1' reference frames
                - x1, y1, z1, x2, y2, z2 - x, y, z coordinates of the N-C1' reference frame (1: left to right, 2: right to left)
                - yaw1, pitch1, roll1, yaw2, pitch2, roll2 - yaw, pitch, roll angles of the N-C1' reference frame
                - euler_phi, euler_theta, euler_psi - euler angles between the N-C1' reference frames (left to right, the other direction is  symmetric)
            * Distance to ideal basepair (if --reference-basepairs is specified)
                - rmsd_edge1, rmsd_edge2 - RMSD of the pairing edge, when fitted on the other bases
                - rmsd_C1N_frames1, rmsd_C1N_frames2 - RMSD of the C1', N1/N9, C6/C8 atoms, when fitted on the other bases
                - rmsd_all_base - "classic" RMSD of all atoms
        When --dssr-binary is specified, DSSR --analyze is executed to gain additional information:
            * dssr_pairing_type - pairing type according to DSSR (e.g. WC, Platform, ~rHoogsteen)
            * pairing_type, shear, stretch, stagger, buckle, propeller, opening, shift, slide, rise, tilt, roll, twist
        """,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="Input file - Parquet with a list of basepairs; FR3D basepairing output; CIF/PDB file (all close contacts will be examined)")
    parser.add_argument("--pdbcache", nargs="+", help="Directories to search for PDB files in order to avoid downloading. Last directory will be written to, if the structure is not found and has to be downloaded from RCSB. Also can be specified as PDB_CACHE_DIR env variable.")
    parser.add_argument("--output", "-o", required=True, help="Output CSV/Parquet file name. Both CSV and Parquet are always written.")
    parser.add_argument("--threads", type=parse_thread_count, default=1, help="Number of worker processes to spawn. Does not affect Polars threading, at the start, the process might use more threads than specified. `0` to use all available CPU threads, `-1` all except one, 50%% for half, ...")
    parser.add_argument("--export-only", default=False, action="store_true", help="Only re-export the input as CSV+Parquet files, do not calculate anything")
    parser.add_argument("--partition-input", default=0, type=int, help="Partition the input by N first characters of the PDBID. Reduces memory usage for large datasets.")
    parser.add_argument("--partition-input-select", default='', type=str, help="Select a given input partition (for example '9' will run pdbids starting with '9').")
    parser.add_argument("--metadata", type=bool, default=True, help="Add deposition_date, resolution and structure_method columns")
    parser.add_argument("--dssr-binary", type=str, help="If specified, DSSR --analyze will be invoked for each structure and its results stored as 'dssr_' prefixed columns")
    parser.add_argument("--filter", default=False, action="store_true", help="Filter out rows for which the values could not be calculated")
    parser.add_argument("--pdb-contact-threshold", default=4.0, type=float, help="When directly loading PDB/mmCIF files, use this as the minimum distance")
    parser.add_argument("--postfilter-hb", default=None, type=float, help="Only include rows with at least 1 hydrogen bond length < X Å")
    parser.add_argument("--postfilter-shift", default=None, type=float, help="Only include rows with abs(coplanarity_shift) < X Å (both left and right)")
    parser.add_argument("--dedupe", default=False, action="store_true", help="Remove duplicate pairs, keep the one with preferred family (W > H > S), FR3D ordering, preferred base order (A > G > C > U), shorter bonds, or lower auth_chain1,auth_nr1")
    parser.add_argument("--reference-basepairs", type=str, help="output of pair_distributions.py with the 'nicest' basepairs, will be used as reference for RMSD calculation (rmsd_edge1, rmsd_edge2, rmsd_C1N_frames1, rmsd_C1N_frames2, rmsd_edge_C1N_frame, rmsd_all_base columns)")
    parser.add_argument("--disable-cross-symmetry", default=False, action="store_true", help="Skip basepairs with bases in different Asymmetrical Units")
    parser.add_argument("--override-pair-family", type=str, required=False, help="Ignore the pair family from the input and assume the specified one instead. Experimental: may specify multiple overrides separated by comma.")
    args = parser.parse_args()

    if args.override_pair_family == "all":
        args.override_pair_family = all_familites

    pdb_utils.set_pdb_cache_dirs(args.pdbcache)

    multiprocessing.set_start_method("spawn")
    if args.threads == 1:
        main(MockPool(), args)
    else:
        with multiprocessing.Pool(processes=args.threads if args.threads > 0 else max(1, os.cpu_count() + args.threads)) as pool:
            main(pool, args)
