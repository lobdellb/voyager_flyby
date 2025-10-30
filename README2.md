# Repository Observations

## High-level purpose
- The existing `README.md` states the goal clearly: convert raw Voyager encounter files into a flyby video. The codebase revolves around downloading, unpacking, processing, and analyzing Voyager Imaging Science Subsystem (ISS) data products.
- Core workflows focus on extracting VICAR image data (`*.IMG` files) and Planetary Data System metadata (`*.LBL` files) from archived tarballs, normalizing the imagery, finding the planet center, and preparing assets for further visualization.

## Project layout
- `src/` contains the application modules. Key entry points and components:
  - `main.py` wires together pipeline tasks for handling Voyager tarballs, persisting metadata, caching converted imagery, and performing circle detection on the planet.
  - `pipeline.py` defines a generic `Pipeline`/`Task` framework and multiple utility classes (e.g., `FileNameParser`).
  - `analysis.py` collects computer-vision helpers (Hough circle detection, image normalization, centering).
  - `helpers.py` houses filename utilities (`extract_stem`, `extract_prefix_from_filename`).
  - `config.py` ensures standard directories (`cache/`, `source_files/`, `output/`) exist and exposes their paths along with a SQLite connection string.
  - `database.py` builds the SQLAlchemy engine/session using the connection string, and configures logging to files under `logs/` (directory expected to exist or be creatable).
  - `models/` contains SQLAlchemy models, currently `VoyagerImage`, mapping a large portion of the Voyager PVL metadata plus local processing artefacts (pickle filenames, circle detection results).
  - `repository/image.py` offers persistence helpers to flatten PVL structures, handle special cases (e.g., `UNK` timestamps), upsert metadata, and look up images by `PRODUCT_ID`.
  - `saturn_flyby_pipeline_notebook.py` appears to be a script/notebook-style module (not yet inspected in detail) likely for exploratory work.
- `sql/` contains raw SQL DDL for the `voyager_images` table and a migration adding circle-detection columns, mirroring the ORM schema.
- `notebooks/` holds several [Marimo](https://marimo.io) notebooks for diagnostics (`diagnose_sqllite_perf.py`), metadata inspection, VICAR issues, Hough tuning, and circle data exploration.
- `tasks.py` defines Invoke tasks for bootstrapping a virtual environment, running the main workflow, and launching the Marimo notebooks. The tasks cache requirement hashes under `.env/.requirements_hash` to skip reinstalling unchanged dependencies.
- `tests/` currently contains a single `pytest` module (`test_helpers.py`) covering `extract_stem` edge cases, including VICAR naming patterns.
- Root-level config includes `pytest.ini`, `requirements.txt`, `requirements-dev.txt`, and licensing.

## Notable classes and functions
- `pipeline.Task`: abstract base for pipeline stages; subclasses must override `process`.
- `pipeline.Pipeline`: orchestrates sequential tasks, feeding the outputs from one into the next, timing each stage, and using `tqdm` for progress reporting.
- `pipeline.FileNameParser`: regex-based parser for Voyager archive member paths (`VGISS_xxx/DATA/Cxxxx_GEOMED.*`).
- `main.ListTarFiles`: yields tarball metadata (path/stem) from the configured `source_path`.
- `main.ExtractTarfile`: indexes tar members, extracts relevant `GEOMED` image and label files into `cache/tar_members/`, and yields dictionaries describing each asset.
- `main.LoadAndStoreMetadata`: batches PVL label ingestion via `repository.image.upsert_image_metadata` and commits every N operations.
- `main.LoadVicarImageToPickle`: loads VICAR image files to Python objects (using `rms-vicar`), caches them as pickles, and updates the database record.
- `main.ComputeAndStoreJupyterCenters`: loads cached images, prepares them with `analysis.scale_image`/`analysis.prep_impage_for_cv2`, runs a parameterized Hough Circle transform, and stores the best fit plus serialized candidates.
- `analysis.find_circle_center_parametrized`/`find_circle_center`: wrappers around OpenCV’s `HoughCircles` tuned for the Voyager imagery.
- `analysis.center_object_in_larger_image` and `analysis.normalize_clip`/`scale_image`: utility functions for aligning and scaling image data prior to visualization.
- `helpers.extract_stem` and `helpers.extract_prefix_from_filename`: essential for deriving IDs from archive naming conventions.
- `repository.image.flatten_vicar_object`: recursively flattens PVL structures, handling nested modules, quantities (value + units), and skipping uninterested keys like `^IMAGE` and `SOURCE_PRODUCT_ID`.
- `repository.image.DatetimeReprEncoder`: intended JSON encoder for datetime logging (currently returns an undefined variable `s` for non-datetime objects—likely a bug worth noting).

## Data flow and persistence
1. Tarballs (`VGISS_*.tar.gz`) placed in `source_files/` are enumerated (`ListTarFiles`).
2. `ExtractTarfile` indexes their members (cached per tar) and extracts `GEOMED` VICAR image/label pairs into `cache/tar_members/`, yielding dictionaries with metadata (stem, suffix, type, etc.).
3. `LoadAndStoreMetadata` opens each `.LBL`, parses PVL metadata, flattens it, and upserts a `VoyagerImage` row (with local filenames for the `.IMG`).
4. `LoadVicarImageToPickle` loads `.IMG` files via `vicar.VicarImage`, saves pickles under `cache/pickled_images/`, and updates the database.
5. `ComputeAndStoreJupyterCenters` loads the pickled images, scales them, runs Hough circle detection, and records best-fit centers plus timing data.
- SQLite database stored at `cache/pipeline.db`; SQLAlchemy engine built with `future=True` flag.
- Additional directories `output/` and potential `cache/member_info/` etc. are managed by the tasks.

## Developer workflows
- **Environment setup:** `invoke create-dev-env` creates `.env/`, installs dependencies from both requirements files when hashes change, and stores a checksum to avoid redundant installs.
- **Running pipeline:** `invoke run-workflow` activates the virtual environment and executes `python src/main.py`.
- **Interactive analysis:** multiple `invoke marimo-*` tasks launch Marimo notebooks for diagnostics and tuning.
- **Testing:** run `pytest` (configured by `pytest.ini`) to execute the helper tests.
- **Database initialization:** `config.py` ensures directories exist; `database.py` (imported in `main.py`) calls `db.Base.metadata.create_all(bind=db.engine)` to materialize tables automatically before pipeline tasks run.

## Observations & quirks
- `config.py` prints the SQLite connection string whenever imported, which can clutter output (e.g., during tests or CLI usage).
- `analysis.py` expects imports like `matplotlib.pyplot as plt` but the snippet shows `plt` used without an explicit import in the captured section; confirm the top of the file includes it (or else this is an oversight).
- `analysis.prep_impage_for_cv2` contains a stray `# print( type( offset_y ) )c` comment ending with `c` and other commented debug statements; overall the module mixes production code with exploratory prints.
- `repository.image.DatetimeReprEncoder.default` references an undefined variable `s`, so JSON encoding for non-datetime objects would raise a `NameError` if executed.
- `tasks.py` relies on `invoke` (provided via `requirements-dev.txt`), and expects to be executed from the repository root (enforced by `check_cwd`).
- SQLAlchemy logging configuration in `database.py` writes to `logs/{logger}.log` but no guard ensures the `logs/` directory exists; developers may need to create it or adjust logging to avoid `FileNotFoundError`.
- Requirements pin fairly recent versions (NumPy 2.2, Matplotlib 3.10, OpenCV 4.12, SQLAlchemy 2.0); confirm compatibility with desired runtime.
- Tests manipulate `sys.path` directly to import from `src/`, implying the package is not installed as a module; running tools that rely on package metadata may require adding `src/` to `PYTHONPATH`.
- Several commented-out sections in `pipeline.py` and `main.py` indicate evolving design ideas (e.g., alternative pipeline database tracking, metadata extraction). They serve as documentation of intent but include stale code.
- The project assumes availability of Voyager ISS tar archives and VICAR support; error handling around missing files or partially processed records is minimal (e.g., raising exceptions when metadata is missing or circle detection fails).
- No continuous integration config is present; manual invocation of `pytest`/`invoke` is implied.

## Potential user operations
- Place Voyager ISS tarballs in `source_files/`, then run `invoke run-workflow` to populate the cache, database, and derived circle data.
- Inspect `cache/pipeline.db` using SQLite tools or provided notebooks for performance and metadata review.
- Open generated pickled images or circle annotations for downstream visualization and video generation (additional scripts likely live elsewhere or are forthcoming).
- Use notebooks to tune detection parameters (`tune_hough_detection.py`) or examine anomalies (`inspect_vicar_problem.py`).
- Export results under `output/` (destination currently prepared by `config.py`, though no scripts shown write there yet).

## Testing & quality
- Minimal automated testing (only filename helper tests). Core pipeline, database, and analysis routines lack coverage.
- No linting or formatting configuration included; style is informal with debug prints and commented code.
- `pytest.ini` likely configures defaults (not inspected but present) to find tests under `tests/`.

## Additional files
- `LICENSE` indicates open-source licensing (details not inspected here but present).
- `CLAUDE.md` may contain supplementary documentation or instructions (contents not reviewed yet).
- `requirements-dev.txt` seems to contain a typo (`invokeroot@...`) in the captured output, suggesting the file might end with `invoke` plus a prompt artifact; verify the actual file contents.

