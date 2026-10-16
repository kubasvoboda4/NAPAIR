# NAPAIR

Nucleic acid basepair processing scripts.

### Setup

* Install Python (3.10 or newer).
* (Recommended) create a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`
* Install the dependencies: `pip install -r requirements.txt`
* Run the scripts with `python`. For example:
    - `python3 ./pairs.py 1EHZ_basepair.txt --output 1ehz_params.csv` - calculates all implemented parameters for all basepairs found by FR3D
    - `python3 ./pairs.py *_basepair.txt --output all_params.csv --pdbcache /downloads/structures/into/here` - same, but for many structures with a cache directory

### Docker

Alternatively, build and run in a container (no local Python setup needed):

```sh
docker build -t napair .
docker run --rm -v "$PWD/work:/work" napair assign_in_structure.py /work/1bna.cif --out /work/out
```

Mount your input/output directory at `/work` and use absolute `/work/...` paths. Any script can be run this way (`pairs.py`, `apply_filter.py`, ...). The image already contains `cdf_files/` and `data/curated/`, so `--validation` works out of the box.

### Usage

All script will output some documentation when started with the `--help` flag.

### List of executable files

Main scripts:

* `assign_in_structure.py`: Annotates basepairs in a given structure (performs the following steps at once)
* `pair_finding.py`: Finds near contacts in PDB structures — outputs any pairs which have atoms closer than 4.0 Å
* `pairs.py`: Annotates pairs with the parameters
* `apply_filter.py`: Filters pairs which satisfy the definitions in parameter-boundaries.csv
* `compare_structures.py`: Compares basepairs of two structures (reference vs. model) and computes PPV, sensitivity, MCC and F1

Other utilities:

* `simulate_fr3d_output.py`: Converts the Parquet table to FR3D-like text files.

### Example commands

All commands should be prefixed with `python` (or `/path/to/venv/bin/python`).

Most script either take path to mmCIF files, or they accept PDB ID and download the files themselves. To avoid repeated downloads, `--pdbcache=/path/to/pdbcache` may be specified. The PDB cache can even be multiple directories — all are used for lookup, but only the last one will be modified if the structure isn't found.

#### `compare_structures.py`

Compares basepairs in a model against a reference structure (MCC = sqrt(PPV * STY), as in RNA-Puzzles). Accepts structure files, PDB IDs or `.csv`/`.parquet` outputs of `assign_in_structure.py`

```sh
compare_structures.py ./reference.cif ./model.cif --list

$ compare_structures.py examples/6UFK-assembly1.cif examples/6UFK-assembly2.cif --chain-map C:A,D:B,F:E
reference: examples/6UFK-assembly1.cif
model:     examples/6UFK-assembly2.cif
chains:    C->A, D->B, F->E (model->reference)
category         TP   FP   FN    PPV    STY    MCC     F1
all              29    2    1  0.935  0.967  0.951  0.951
canonical        22    1    0  0.957  1.000  0.978  0.978
non-canonical     7    1    1  0.875  0.875  0.875  0.875
```

#### `assign_in_structure.py`

Annotate pairs in a given structure (output all supported file formats)

```sh
assign_in_structure.py ./examples/6ROS.cif --out ./outdir/
```

Annotates pairs in a large set of structures

```sh
assign_in_structure.py ~/pdb/**/*.cif --out ./outdir/ --threads=100% --format=parquet
```

Annotates specific basepair in given structure

```sh
assign_in_structure.py  ./example/6ROS.cif ./example--out ./outdir/ --basepair_id A_2_A_17
```

Annotates and validates pairs in given structure

```sh
assign_in_structure.py ./example/6ROS.cif --out ./outdir/ --validation
```

#### `pair_finding.py`

Find all contacts under 4.0 Å in a large set of structures - given by PDB IDS

```sh
python pair_finding.py $(cat ./pdbids.lines) --pdbcache ~/pdbcache --output ./pairf-output/ --distance 4.0 --threads 100%
```

#### `pairs.py`

Annotate basepairs from FR3D using our parameters. Please, run FR3D with the `-c basepair_detail` option. `--reference-basepairs` is an optional list of example pairs used for RMSD calculation.

```bash
pairs.py ./fr3d-output/*_basepair_detail.txt \
    --pdbcache ~/pdbcache \
    --threads 100% \
    --dedupe \ # only keep one orientation of each pair
    --reference-basepairs ./reference-basepairs.json \ # optional
    --output ./out/fr3d-pairs.parquet
```

Annotate pairs from pair_finding.py. 

```bash
pairs.py\
    --override-pair-family=all\ # pairs do not have a family column, we want to calculate parameters for each pair in each family
    --filter --dedupe\
    --postfilter-hb=4.2 --postfilter-shift=2.5\ # require that all pairs have at least one 4.2 Å H-bond, and |shift| < 2.5Å
    --pdbcache ~/pdbcache\
    --threads=100%\
    --output=./out/all.parquet\
    pairf-output/*.parquet\
    --partition-input-select=0-32/32 # run in batches of N/32 to lower memory usage
```

#### `apply_filter.py`

Apply the filter based on the class definitions in `parameter-boundaries.csv`.
The `--best-fit=single-pair` is used to select only one classification for each pair of nucleotides.
The parameter is optional (or can be specified `=none`) to avoid this filtering step.

```bash
apply_filter.py out/all.parquet -o "out/assignment.parquet" --best-fit=single-pair
```

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.
