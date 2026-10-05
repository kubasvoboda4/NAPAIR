import functools
import itertools
import os, math, re, csv, dataclasses
from typing import Literal, Optional, Union

@functools.total_ordering
@dataclasses.dataclass(frozen=True)
class PairType:
    """
    Represent basepairing class, such as cWW-A-U, cWWa-U-U, ntHS-A-G, ...
    """
    type: str # eg cWS or tHH
    bases: tuple[str, str]
    subclass: str

    def __post_init__(self):
        if len(self.type) != 3 or len(self.bases) != 2 or self.type[0] not in "ct" or self.type[1].upper() not in "BWSH" or self.type[2].upper() not in "WSHB" or (len(self.bases[0]) == 1 and self.bases[0] not in "ACGU") or (len(self.bases[1]) == 1 and self.bases[1] not in "ACGU"):
            raise ValueError(f"Invalid pair type: PairType({repr(self.type)}, {self.bases}), {self.subclass}")
    @property
    def full_type(self) -> str:
        """DEPRECATED"""
        return f"{self.type}{self.subclass}"
    @property
    def full_family(self) -> str:
        """Basepair family including the subclass"""
        return f"{self.type}{self.subclass}"
    @property
    def bases_str(self) -> str:
        """Pairing bases separated by dash (G-C, A-U, ...)"""
        return "-".join(self.bases)
    def __str__(self) -> str:
        return f"{self.full_type}_{self.bases_str}_{self.subclass}"
    def __repr__(self) -> str:
        return f"PairType({repr(self.type)}, {self.bases}{f', {repr(self.subclass)}' if self.subclass else ''})"
    def to_tuple(self, simplify = False) -> tuple[str, str]:
        """Tuple of pair family and bases. E.g. ('cWW', 'A-U')"""
        return (self.type if simplify else self.full_type, self.bases_str)
    def swap(self) -> 'PairType':
        """Swap the bases and pairing edges. E.g. cWH-A-U -> cHW-U-A"""
        return PairType(self.type[0] + self.type[2] + self.type[1], (self.bases[1], self.bases[0]), self.subclass)
    def is_preferred_orientation(self) -> bool:
        """cWH > cHW, cWS > cSW, ... A-U > U-A, G-C > C-G, ..."""
        return is_preferred_pair_type_orientation(self.to_tuple())
    def is_swappable(self):
        """If this pair type can be safely swapped, i.e. the other orientatin isn't defined to mean something different."""
        if self.to_tuple() in hbonding_atoms and self.swap().to_tuple() in hbonding_atoms:
            # both definitions exist,
            return False
        return True
    def swap_is_nop(self) -> bool:
        """If the swap is the same as the original pair type."""
        return self.type[1] == self.type[2] and self.bases[0] == self.bases[1]
    def order_key(self):
        t = self.type.lower()
        return (
            pair_families.index(t) if t in pair_families else 1000,
            t,
            self.bases[1],
            self.bases[0],
            self.subclass,
        )
    @property
    def edge1_name(self) -> Literal['W', 'H', 'S', 'B']:
        assert len(self.type) == 3
        return self.type[1].upper() # type: ignore
    @property
    def edge2_name(self) -> Literal['W', 'H', 'S', 'B']:
        assert len(self.type) == 3
        return self.type[2].upper() # type: ignore
    
    def __lt__(self, other: 'PairType') -> bool:
        return self.order_key() < other.order_key()
    def __eq__(self, other: 'PairType') -> bool:
        return self.order_key() == other.order_key()
    def __hash__(self) -> int:
        return hash(self.order_key())
    @staticmethod
    def from_tuple(t: Union['PairType', tuple[str, str, str]]) -> "PairType":
        if isinstance(t, PairType):
            return t
        type, bases, subclass = t
        return PairType.create(type, *bases.split("-"), subclass) # type: ignore
    @staticmethod
    def create(type, base1, base2, subclass, name_map: Optional[dict[str, str]] = None):
        if name_map is not None:
            base1 = name_map.get(base1, base1)
            base2 = name_map.get(base2, base2)
        return PairType(type, (base1, base2), subclass)
    @staticmethod
    def parse(s: str, normalize_case=True) -> "PairType":
        family_regex = r"(?P<n>n)?(?P<cistrans>[ct])(?P<family>[WHSB][WHSB])"
        bases_compact_regex = r"(?P<base1a>[ATGCU])(?P<base2a>[ATGCU])"
        bases_full_regex = r"(?P<base1b>\w+)-(?P<base2b>\w+)"
        subclass_regex = r"(?P<subclass>[0-9]+)"
        if (m := (re.match(f"^{family_regex}(-|_|\\s+)({bases_compact_regex}|{bases_full_regex})(-|_|\\s+){subclass_regex}$", s, re.IGNORECASE) or
                 re.match(f"^{bases_full_regex}-{family_regex}-{subclass_regex}$", s, re.IGNORECASE)
            )):
            cistrans = m.group("cistrans")
            family_core = m.group("family")
            subclass = m.group("subclass")
            if normalize_case:
                cistrans = cistrans.lower()
                family_core = family_core.upper()
            type = cistrans + family_core
            base1: str = m.groupdict().get("base1a") or m.groupdict().get("base1b") or ""
            base2: str = m.groupdict().get("base2a") or m.groupdict().get("base2b") or ""
            return PairType(type, (base1, base2), subclass)
        raise ValueError(f"Invalid pair type: {s}")
    
    def normalize_capitalization(self) -> 'PairType':
        """Normalize the capitalization of pair family lower t/c, twice upper W/H/S, optonal lower a suffix"""
        if self.type[0].islower() and self.type[1].isupper() and self.type[2].isupper():
            return self
        else:
            t = self.type[0].lower() + self.type[1].upper() + self.type[2].upper()
            return PairType(t, self.bases, self.subclass)

resname_map = {
    'DT': 'U',
    'DC': 'C',
    'DA': 'A',
    'DG': 'G',
    'DU': 'U',
    'T': 'U',
    'BRU':'U',
    '5MC':'C',
    '8OG':'G',
    '3ZO':'G',
    'PSU':'U',
    'OMC':'C',
    '5MU':'U',
    'OMU':'U',
    'MA6':'A',
    '2MG':'G',
    'UR3':'U',
    '6MZ':'G',
    'G7M':'G',
    '4OC':'U',
    '2MA':'A',
    'H2U':'U',
    '4SU':'U',
    '7MG':'G',
    '1MG':'G',
    '1MA':'A',
}
def map_resname(resname: str) -> str:
    """Replaces DX with X, and T with U"""
    return resname_map.get(resname.upper(), resname)
# Maps nonstandard residues' atom names to the canonical atom names they replace.
# Direction: canonical_atom_name -> actual_atom_name_in_residue.
atom_name_map: dict[str, dict[str, str]] = {
    '3ZO': {'N2': 'O2'},
    '4SU': {'O4': 'S4'},
    'PSU': {'O4': 'O2', 'O2': 'O4','C5': 'N1', 'N1': 'C5', 'C4': 'C2', 'C2': 'C4', 'N1': 'C5'}, 
}

def map_atom_name(resname: str, canonical_atom: str) -> str:
    """Returns the actual atom name to look up in `resname` for a given canonical atom name."""
    return atom_name_map.get(resname.upper(), {}).get(canonical_atom, canonical_atom)

def read_pair_definitions(file = os.path.join(os.path.dirname(__file__), "H_bonding_Atoms_from_Isostericity_Table.csv")) -> dict[tuple[str, str], list[tuple[str, str, str, str]]]:
    """Load the hydrogen bond definitions from H_bonding_Atoms_from_Isostericity_Table.csv file into the `hbonding_atoms` dict."""
    with open(file, "r") as f:
        reader = csv.reader(f)
        lines = list(reader)
    header_i = next(i for i, line in enumerate(lines) if len(line) > 3 and line[0] == 'TYPE')
    header = lines[header_i]
    data = lines[header_i + 1:]
    def translate_pair_type(line):
        m = re.match(r"^(cis |trans )([WSH])/([WSH])", line[0].strip(), re.IGNORECASE)
        if not m:
            print("WARNING: Invalid pair type: " + line[0])
            return None
        ct = { "c": "c", "t": "t", "cis":"c", "trans": "t" }[m.group(1)[0].strip().lower()]
        return f"{ct}{m.group(2).upper()}{m.group(3).upper()}"
    
    result_mapping = {}
    for line in data:
        try:
            res1, res2 = line[1].strip(), line[2].strip()
            subclass = line[6]
            pt = (translate_pair_type(line), res1 +"-"+ res2, subclass)
            if pt[0] is None:
                continue
            if pt not in result_mapping:
                result_mapping[pt] = []
            assert line[4] == '---', line
            atom1, atom2 = line[3].strip(), line[5].strip()
            b = ("A"+get_angle_ref_atom(res1, atom1), "A"+atom1, "B"+atom2, "B"+get_angle_ref_atom(res2, atom2))
            if b not in result_mapping[pt]: 
                result_mapping[pt].append(b)
        except Exception as e:
            print("ERROR on line: ", line)
            raise
    return result_mapping

def get_angle_ref_atom(res: str, atom: str)-> str:
    """
    Gets the reference atom used to calculate the hydrogen bond donor/acceptor angle.
    """
    res = map_resname(res)
    neighbors = set(itertools.chain(*[ edge for edge in atom_connectivity[res] if atom in edge ]))
    neighbors.remove(atom)
    assert len(neighbors) >= 1, f"No neighbors found in {res} for atom {atom}"
    return max(neighbors)

def hbond_swap_nucleotides(hbond: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
    """
    Replace A with B and B with A in the hydrogen bond definition.
    """
    return tuple(
        atom.replace("A", "ⒷⒷⒷⒷⒷⒷ").replace("B", "A").replace("ⒷⒷⒷⒷⒷⒷ", "B")
        for atom in hbond
    ) # type: ignore

def _swap_pair_type(pair_type: tuple[str, str, str]) -> tuple[str, str, str]:
    t, bases, subclass = pair_type
    bases = bases.split("-")
    assert len(pair_type) == 3 and len(t) >= 3 and len(bases) == 2
    assert t[0] in "ct"
    return (t[0] + t[2] + t[1] + t[3:], bases[1] + "-" + bases[0], subclass)

def is_preferred_pair_type_orientation(pair_type: Union[PairType, tuple[str, str, str]]) -> bool:
    if isinstance(pair_type, tuple):
        pair_type = PairType.from_tuple(pair_type)
    other = pair_type.swap()
    if other == pair_type:
        return True
    if other.type == pair_type.type:
        # symmetric pair type like cWW, cHH, ...
        # we prefer A > G > C > U (who knows why)
        preference = [ "A", "DA", "G", "DG", "C", "DC", "U", "DU", "T", "DT" ]
        bases = pair_type.bases
        if bases[0] not in preference and bases[1] not in preference:
            # weird bases, alphabetical order
            return bases[0] <= bases[1]
        elif bases[0] not in preference:
            return False
        elif bases[1] not in preference:
            return True

        return preference.index(bases[0]) <= preference.index(bases[1])
    else:
        # non-symmetric pair type like cWH, cWS, ...
        return pair_type.type.lower() in pair_families

# all pair type ordered according to The Paper
pair_families = [
    "cww", "tww",
    "cwh", "twh",
    "cws", "tws",
    "chh", "thh",
    "chs", "ths",
    "css", "tss",
    "cwb"
]

pair_families_ids = { pt: ix + 1 for ix, pt in enumerate(pair_families) }

sugar_atom_connectivity = [
    ("P", "OP2"), ("P", "OP1"), ("P", "O5'"),
    ("O5'", "C5'"), ("C5'", "C4'"),
    ("C4'", "O4'"), ("C4'", "C3'"),
    ("C3'", "O3'"), ("C3'", "C2'"),
    ("C2'", "O2'"), ("C2'", "C1'"),
    ("C1'", "O4'")
]
purine_atom_connectivity = [
    *sugar_atom_connectivity,
    ("C1'", "N9"), ("N9", "C8"), ("C8", "N7"),
    ("N9", "C4"), ("N7", "C5"),
    ("C4", "C5"), ("C5", "C6"), ("C6", "N1"), ("N1", "C2"), ("C2", "N3"), ("N3", "C4")
]
pyrimidine_atom_connectivity = [
    *sugar_atom_connectivity,
    ("C1'", "N1"), ("N1", "C2"), ("C2", "N3"),
    ("N3", "C4"), ("C4", "C5"), ("C5", "C6"),
    ("N1", "C6"),
]
atom_connectivity = {
    "A": [
        *purine_atom_connectivity,
        ("C6", "N6")
    ],
    "G": [
        *purine_atom_connectivity,
        ("C6", "O6"), ("C2", "N2")
    ],
    "C": [
        *pyrimidine_atom_connectivity,
        ("C2", "O2"),
        ("C4", "N4"),
        # ("C5", "C7") # 5-methyl cytosine
    ],
    "U": [
        *pyrimidine_atom_connectivity,
        ("C2", "O2"),
        ("C4", "O4")
    ]
}
atom_connectivity["T"] = [ *atom_connectivity["U"], ("C5", "C7") ]

base_edges = {
    "A": {
        "S": [ "O2'", "N3", "C2" ],
        "W": [ "C2", "N1", "N6" ],
        "H": [ "N6", "N7", "C8" ], # C5
    },
    "G": {
        "S": [ "O2'", "N3", "N2" ],
        "W": [ "N2", "N1", "O6" ],
        "H": [ "O6", "N7", "C8" ],
    },
    "T": {
        "S": [ "O2'", "O2" ],
        "W": [ "O2", "N3", "O4" ],
        "H": [ "O4", "C7" ],
    },
    "U": {
        "S": [ "O2'", "O2" ],
        "W": [ "O2", "N3", "O4" ],
        "H": [ "O4", "C5" ],
    },
    "C": {
        "S": [ "O2'", "O2" ],
        "W": [ "O2", "N3", "N4" ],
        "H": [ "N4", "C5" ],
    }
}

hbonding_atoms = read_pair_definitions()

def is_ch_bond(pair_type: PairType, b: tuple[str, str, str, str]) -> bool:
    """Does the H-bond involve a carbon atom?"""
    return b[1][1] == 'C' or b[2][1] == 'C'

def is_bond_to_sugar(pair_type: PairType, b: tuple[str, str, str, str]) -> bool:
    """Is the bond to the O2' (or other sugar) atom?"""
    return b[1].endswith("'") or b[2].endswith("'")

def is_bond_hidden(pair_type: PairType | tuple, b: tuple[str, str, str, str]) -> bool:
    """Return true to hide a given bond from the generated plots"""
    return False
    return is_ch_bond(pair_type, b)
    return is_ch_bond(pair_type, b) or is_bond_to_sugar(pair_type, b)

def get_hbonds(pair_type: Union[tuple[str, str, str], PairType], throw=True) -> list[tuple[str, str, str, str]]:
    """Returns the list of hydrogen bonds for the given pair type. If it isn't defined, return an empty list or throw"""
    if isinstance(pair_type, PairType):
        type = pair_type.type
        subclass = pair_type.subclass
        bases = pair_type.bases_str
    else:
        type, bases, subclass = pair_type

    if type[1].islower() or type[2].islower():
        type = type[0] + type[1].upper() + type[2].upper()
    if (type, bases, subclass) in hbonding_atoms:
        return hbonding_atoms[(type, bases, subclass)]
    
    (type, bases, subclass) = _swap_pair_type((type, bases, subclass))
    if (type, bases, subclass) in hbonding_atoms:
        return [ hbond_swap_nucleotides(hbond) for hbond in hbonding_atoms[(type, bases, subclass)] ]
    
    if throw:
        raise KeyError(f"Pair type {pair_type} not found in hbonding_atoms")
    else:
        return []
    
def has_symmetrical_definition(pair_type: PairType):
    """Is the hydrogen bond definition of a swappable pair type symmetrical? - i.e. tWW A-A is the same regardless if one base is left or right."""
    if not pair_type.swap_is_nop():
        return False
    hb = set(get_hbonds(pair_type))
    hb_swap = set(hbond_swap_nucleotides(x) for x in hb)
    return hb == hb_swap

def defined_pair_types() -> list[PairType]:
    return [ PairType.from_tuple(k) for k in hbonding_atoms.keys() ]


