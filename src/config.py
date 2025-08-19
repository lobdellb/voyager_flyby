import pathlib

repo_root = pathlib.Path(__file__).parent.parent

cache_path = repo_root / "cache"
if not cache_path.exists():
    cache_path.mkdir(parents=True)

source_path = repo_root / "source_files"
if not source_path.exists():
    source_path.mkdir(parents=True)

output_path = repo_root / "output"
if not output_path.exists():
    output_path.mkdir(parents=True)

db_path = cache_path / "pipeline.db"

db_loaded_fn = cache_path / "db_loaded.lock"

db_conn_str = f"sqlite:///{db_path}"

print( db_conn_str)