# ChaosNet: Stage 0

Stage 0 establishes the repository, shared contracts, reproducible fake data, and
smoke test for the reset-and-retrain chaos prediction and feedback control project.
The scientific implementation starts in later stages.

## Install and verify

Use Python **3.11** in a new virtual environment. Direct dependencies are pinned
in `requirements.txt`; `pyproject.toml` reads that file so both install paths use
the same versions. CPU PyTorch wheels are selected directly from the official
PyTorch host on Windows AMD64 and Linux x86-64. macOS ARM64 uses the native
PyPI distribution, which has no CUDA runtime. Those are the supported install
platforms for this Stage 0 dependency set.

PowerShell, from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m chaosnet smoke
```

If the Windows Python launcher is unavailable, replace `py -3.11` with the full
path to a Python 3.11 executable. No activation or execution-policy change is
needed. A project-local verification interpreter, when present, can be used as:

```powershell
.\.tools\python\python.exe -m venv .venv
```

Linux/macOS:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q
.venv/bin/python -m chaosnet smoke
```

An alternative dependency installation is `python -m pip install -r requirements.txt`
followed by `python -m pip install -e .`. It is unnecessary for the editable
installation above, which installs the dependencies itself. Internet access to
PyPI and `download.pytorch.org` is required on the first installation.

The smoke command prints exactly `OK` on success. To check Python 3.11, virtual
environment use, CPU PyTorch, dependency consistency, pytest, and the smoke command
together, run this with the virtual environment's Python:

```sh
python scripts/check_stage0.py
```

## Repository tree

```text
chaos/
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- chaosnet/
|   |-- __init__.py
|   |-- __main__.py
|   |-- _serialization.py
|   |-- systems/
|   |   |-- __init__.py
|   |   `-- base.py
|   |-- datagen/
|   |   |-- __init__.py
|   |   `-- fake.py
|   |-- data/
|   |   |-- __init__.py
|   |   |-- schema.py
|   |   `-- validate.py
|   |-- model/__init__.py
|   |-- session/__init__.py
|   |-- control/__init__.py
|   |-- eval/__init__.py
|   `-- configs/
|       |-- __init__.py
|       |-- loader.py
|       `-- example.yaml
|-- tests/
|   |-- conftest.py
|   |-- test_schema.py
|   |-- test_fake.py
|   |-- test_config.py
|   |-- test_adapter.py
|   `-- test_smoke.py
`-- scripts/
    `-- check_stage0.py
```

`tests/` and `scripts/` live at the repository root; the eight runtime packages
live under `chaosnet/`. Generated environments, build metadata, and caches are
ignored by Git. The later-stage package directories contain only package
documentation, with no unimplemented callable functions.

## Frozen data contracts

`chaosnet.data.schema.SystemSpec` is a frozen dataclass with exactly these fields,
in architecture order:

```text
name, state_names, state_units, param_names, ext_names, ctrl_names,
ctrl_bounds, ctrl_rate_limit, Ts, window, horizon_s, lyap_threshold,
undesirable=("chaotic",), notes={}
```

`notes` has an independent default dictionary per instance. As specified by the
architecture's frozen dataclass, nested dictionaries are not deeply immutable;
contributors must treat a specification's notes as read-only after creation.

| Property | Definition |
| --- | --- |
| `n_state` | `len(state_names)` |
| `n_ext` | `len(ext_names)` |
| `n_ctrl` | `len(ctrl_names)` |
| `n_seq` | `n_state + n_ext + n_ctrl` |

The two-state, three-external-signal example has sequence width 6. A three-state
system with one external signal and one actuator has width 5, correcting the
inconsistent Chua table entry in the PDF using its stated dimension formula.

`to_dict()` produces independent JSON-compatible data; `from_dict()` restores the
tuple fields. Hashing uses SHA-256 over UTF-8 JSON with sorted keys, compact
separators, and finite values only. It does not use Python's process-dependent
`hash()`. Notes and metadata use string keys and JSON-compatible values.

`RunRecord` has exactly the agreed fields:

```text
run_id, params, t, X, ext, u, regime, lyap, mode, meta
```

| Array | Shape |
| --- | --- |
| `t` | `(T,)` |
| `X` | `(T, n_state)` |
| `ext` | `(T, n_ext)` |
| `u` | `(T, n_ctrl)` |
| `regime` | `(T,)`, conventionally int8, 0 non-chaotic / 1 chaotic |
| `lyap` | `(T,)` |

`params` contains static physical parameters. `mode` is `open_loop`,
`expert_controlled`, or `candidate_trial`. `u[k]` is the applied command for sample
interval k; a future window builder uses the preceding applied commands.

`run.save_npz(path)` and `RunRecord.load_npz(path)` preserve numeric array values,
shapes, dtypes, and JSON metadata. NPZ loading uses `allow_pickle=False`. The parent
directory must already exist; saving writes the exact supplied filename.

## Shared interfaces

```python
adapter.spec(cfg)                         # -> SystemSpec
adapter.rhs(t, x, p, e, u)                 # -> derivative ndarray
adapter.ext_signals(t, params, cfg)        # -> external-signal ndarray
adapter.candidate_controllers()           # -> dict of controller factories
make_fake_runs(spec, n_runs, seed)         # -> list[RunRecord]
validate(runs, spec)                      # -> {"n_runs": int, "n_samples": int}
load_cfg(path)                            # -> dict
config_hash(cfg)                          # -> SHA-256 hexadecimal string
```

`SystemAdapter` is abstract: a concrete adapter must supply all four methods.
Adapter arrays carry channels on their final axis; preceding axes can hold a
batch. Parameters are mappings with scalar or broadcastable array values.

Fake runs use a local `numpy.random.Generator`, carry `meta["synthetic"] = True`,
and are long enough for a complete window and future horizon. They provide no
evidence about real dynamics or chaos. The Stage 0 validator checks shapes,
finite values, parameter names, sampling intervals, run modes, and unique IDs.
It does not assess numerical simulation accuracy or label correctness.

## Five-member handoff

| Member | Interface available for development |
| --- | --- |
| M1: systems and datasets | `SystemAdapter`, `SystemSpec`, `RunRecord`, NPZ I/O |
| M2: model and prediction | Derived tensor widths and reproducible fake runs |
| M3: preprocessing and session | Serialization, stable hashes, validation, fake runs |
| M4: control and validation | Actuator metadata, adapter signatures, applied-control convention |
| M5: integration and evaluation | Pinned installation, config loader/hash, CLI, tests, check script |

The test suite covers the exact contract fields, frozen specification attributes,
JSON and NPZ round-trips, hash order independence and cross-process consistency,
both system widths, fake-data reproducibility, unchanged global RNG state,
invalid data rejection, abstract adapter signatures, and the installed smoke
command from outside the checkout.

Stage 0 deliberately ends at these interfaces. Duffing/Chua equations, numerical
integrators, ChaosCtrlNet, training, sessions/reset behavior, controllers, and
scientific evaluation belong to subsequent stages.

## Verification record

Verified on Windows AMD64 on 4 November 2026, using a newly created Python 3.11.18
virtual environment with system site packages disabled:

- `python -m pip install -e .`: succeeded and installed the pinned dependencies.
- All eight agreed dependencies imported successfully.
- PyTorch reported `2.5.1+cpu` with `torch.version.cuda is None`.
- `python -m pip check`: no broken requirements found.
- `pytest -q`: **26 passed**, with no warnings in the final run.
- `python -m chaosnet smoke`: **OK**.

Environment issues resolved during verification: Python 3.11 was not initially
installed, so a portable runtime was placed in ignored `.tools/python/`. The first
sandboxed test run could not access pytest's temporary directory; the tests passed
with normal user permissions. Its generated cache was cleared to remove the
resulting cache-permission warnings. No tests were skipped or disabled.

The source contracts and local repository are ready for all five contributors.
Linux and macOS installation markers are configured but were not tested here.
