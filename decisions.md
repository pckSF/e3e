# Trajectory Data Storage: Format Decision

## Problem Statement

We collect RL trajectory data (observations, actions, policy_logits, rewards,
next_observations, terminals, truncated) step-by-step from a Gymnasium
environment. The dataset shape is `n_episodes × max_steps` per field, with
varying feature dimensions (e.g. observations are `[T, obs_dim]`, actions are
`[T]`).

**Constraints:**
- Cannot accumulate all data in memory (OOM when stacking JAX arrays)
- Must write incrementally (per-timestep or per-episode)
- Must support efficient reads for downstream ML training
- Dataset sizes will grow significantly beyond the current 1,000 episodes
- Fields have different shapes/dtypes (float observations vs int actions vs bool terminals)

---

## Options Considered

### 1. CSV (current approach)

**How it works:** Each field written to a separate `.csv` file, one row per
timestep, appended via `csv.writer`.

| Pros | Cons |
|------|------|
| Human-readable | ~10× larger on disk than binary formats |
| No extra dependencies | No type information (everything is strings) |
| Simple append | Very slow to read back for training |
| | No compression |
| | No partial/random-access reads |
| | Parsing overhead on load (string→float conversion) |
| | Separate files per field — no atomic consistency |

**Verdict:** Adequate for debugging small runs but does not scale. Reading back
250k+ rows of float CSVs into arrays is painfully slow and memory-hungry.

---

### 2. HDF5 (h5py)

**How it works:** Hierarchical binary format. Each field is a *dataset* inside
an HDF5 file. Datasets can be created with `maxshape=(None, ...)` and extended
incrementally via `dataset.resize()` + slice assignment.

| Pros | Cons |
|------|------|
| **Industry standard for offline RL data** (D4RL, Minari, RoboNet) | Single-writer model (no concurrent appends from multiple processes) |
| Chunked storage → constant-memory writes | Possible file corruption on hard crash mid-write |
| Built-in compression (gzip, lzf) | Slightly more complex API than pickle |
| Partial / random-access reads by index | |
| Native NumPy array support | |
| Stores metadata (attrs) alongside data | |
| Already declared in `pyproject.toml` | |
| Mature, battle-tested (20+ years) | |
| Single file per dataset — atomic & portable | |

**Memory profile for writes:**
```
per-step cost ≈ chunk_size × element_bytes  (typically < 1 MB)
```
Completely independent of total dataset size.

**Verdict:** Strong match. The offline RL community has converged on HDF5 — D4RL
and its successor Minari both use it. Supports exactly the incremental-write,
random-access-read pattern we need.

---

### 3. Parquet (via Polars)

**How it works:** Columnar binary format, optimized for tabular analytics.
Polars (already in `pyproject.toml`) can write Parquet natively.

| Pros | Cons |
|------|------|
| Excellent compression (Snappy, Zstd) | Designed for tabular data, not N-dimensional arrays |
| Fast columnar reads | Multi-dim fields (obs, logits) must be flattened or stored as list columns |
| Schema enforcement | Append workflow is awkward (write separate files, then concat) |
| Good ecosystem (DuckDB, Spark, Polars) | No true in-place append — must write row-groups or separate files |
| Already a dependency (`polars`) | Overkill for simple numeric arrays |

**Verdict:** Great for tabular analytics, but RL trajectory data is
fundamentally array-oriented, not tabular. Flattening `obs_dim=8` observations
into 8 separate columns or storing them as list-columns adds unnecessary
complexity. Parquet shines for heterogeneous, schema-rich data — not for
homogeneous numeric tensors.

---

### 4. NumPy memory-mapped files (np.memmap)

**How it works:** Pre-allocate a binary file on disk with a known shape. Write
into it via array indexing. Reads are zero-copy via OS page cache.

| Pros | Cons |
|------|------|
| Zero-copy reads (OS-managed) | Must know total shape upfront (or over-allocate) |
| Zero dependencies | No compression |
| Extremely fast sequential reads | One file per array — no metadata, no grouping |
| Simple API | Resizing requires creating a new file |
| | No built-in chunking control |

**Verdict:** Good performance but inflexible. Since episode count will grow and
we don't know final sizes upfront, the fixed-shape requirement is a significant
limitation. Also no compression or metadata support.

---

### 5. Zarr

**How it works:** Chunked, compressed N-dimensional arrays. Similar to HDF5 but
more Pythonic API, supports cloud backends (S3, GCS).

| Pros | Cons |
|------|------|
| Resizable arrays with chunking | Extra dependency (not in project) |
| Good compression | Less established in RL community |
| Cloud-native storage backends | Stores as directory of chunk files (many small files) |
| Pythonic API | |

**Verdict:** Technically capable but adds a dependency and is not the community
standard for offline RL data. If cloud storage were needed this would be worth
reconsidering.

---

### 6. Pickle with chunked writes

**How it works:** Periodically pickle batches of episodes to separate files,
then concatenate at the end or load lazily.

| Pros | Cons |
|------|------|
| No extra dependencies | No partial reads (must deserialize entire chunk) |
| Simple | Security risk (arbitrary code execution on load) |
| Fast writes | No compression |
| | Multiple files to manage |
| | Fragile across Python/NumPy version changes |

**Verdict:** Pickle is fine for checkpointing model state (as Orbax does) but
poor for large datasets that need random access and will be loaded repeatedly.

---

## Decision Matrix

| Criterion | Weight | CSV | HDF5 | Parquet | memmap | Zarr | Pickle |
|---|---|---|---|---|---|---|---|
| Incremental writes (constant memory) | High | ✓ | ✓ | ○ | ○ | ✓ | ○ |
| Random-access reads | High | ✗ | ✓ | ○ | ✓ | ✓ | ✗ |
| Compression | Medium | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ |
| Community standard (offline RL) | High | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Already a dependency | Low | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| Multi-dim array support | High | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| Metadata support | Low | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ |
| Single-file output | Medium | ✗ | ✓ | ✗ | ✗ | ✗ | ○ |

Legend: ✓ = strong, ○ = partial, ✗ = weak

---

## Decision: HDF5 via h5py

**Primary reasons:**
1. **Industry standard** — D4RL, Minari (Farama Foundation), and most offline RL
   research use HDF5. Following this convention makes the data immediately
   compatible with standard tooling and recognizable to anyone in the field.
2. **Constant-memory incremental writes** — Create datasets with
   `maxshape=(None, ...)`, then `resize()` + slice-assign per episode/step.
   Memory usage stays flat regardless of dataset size.
3. **Efficient random-access reads** — During training, load only the slices you
   need. With chunking, only relevant chunks are read from disk.
4. **Compression** — `gzip` or `lzf` compression reduces disk usage 2-5× for
   float trajectory data with no API complexity.
5. **Already declared** — `h5py>=3.16.0` is in `pyproject.toml`.

---

## Re-evaluation

After reviewing the full analysis, the decision holds. The only serious
alternative would be **Zarr** (modern, cloud-native, similar capabilities), but:
- Not an existing dependency
- Not the offline RL community convention (D4RL/Minari both chose HDF5)
- Directory-of-files storage is more fragile than a single `.hdf5` file

**Parquet/Polars** is tempting since it's a dependency, but fundamentally
mismatched — it's a columnar tabular format, and our data is N-dimensional
numeric arrays. Forcing arrays into columns adds complexity with no benefit.

**Decision confirmed: HDF5.**

Legend: ✓ = strong, ○ = partial, ✗ = weak

---

## Decision: HDF5 via h5py

**Primary reasons:**
1. **Industry standard** — D4RL, Minari (Farama Foundation), and most offline RL
   research use HDF5. Following this convention makes the data immediately
   compatible with standard tooling and recognizable to anyone in the field.
2. **Constant-memory incremental writes** — Create datasets with
   `maxshape=(None, ...)`, then `resize()` + slice-assign per episode/step.
   Memory usage stays flat regardless of dataset size.
3. **Efficient random-access reads** — During training, load only the slices you
   need. With chunking, only relevant chunks are read from disk.
4. **Compression** — `gzip` or `lzf` compression reduces disk usage 2-5× for
   float trajectory data with no API complexity.
5. **Already declared** — `h5py>=3.16.0` is in `pyproject.toml`.

**Implementation plan:**
- Write a `TrajectoryWriter` context manager that opens an HDF5 file, creates
  extensible datasets, and provides an `add_timestep()` method
- Write data per-episode (batch all steps within one episode, extend datasets
  once per episode) to minimize resize calls
- Store metadata (env name, seed, checkpoint path, episode count) as HDF5 attrs
- Add a `TrajectoryReader` or extend `TrajectoryData.load()` for HDF5 reads
- Update `collect_data()` to use the new writer instead of `DataLogger`
