import pathlib



cache_path = pathlib.Path(__file__).parent / "cache"
if not cache_path.exists():
    cache_path.mkdir(parents=True)

source_path = pathlib.Path(__file__).parent / "source"
if not source_path.exists():
    source_path.mkdir(parents=True)

output_path = pathlib.Path(__file__).parent / "output"
if not output_path.exists():
    output_path.mkdir(parents=True)

db_path = cache_path / "pipeline.db"