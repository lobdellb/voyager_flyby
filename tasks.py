from invoke import task


import os
import hashlib

dev_env = ".env"
env_activate_fn = f"./{dev_env}/bin/activate"
hash_file = f"{dev_env}/.requirements_hash"


def get_file_hash(filename):
    """Generate MD5 hash of a file's contents."""
    if not os.path.exists(filename):
        return ""
    with open(filename, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_requirements_hash():
    """Generate combined hash of both requirements files."""
    req_hash = get_file_hash("requirements.txt")
    dev_hash = get_file_hash("requirements-dev.txt")
    return hashlib.md5((req_hash + dev_hash).encode()).hexdigest()

def get_stored_hash():
    """Read the stored hash from the hash file."""
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            return f.read().strip()
    return ""

def store_hash(hash_value):
    """Store the hash value to the hash file."""
    os.makedirs(dev_env, exist_ok=True)
    with open(hash_file, 'w') as f:
        f.write(hash_value)


def check_cwd():

    if os.path.dirname(os.path.abspath(__file__)) != os.getcwd():
        raise Exception("Please run this script from the repository root directory.")


@task
def create_dev_env(c):

    check_cwd()

    if not os.path.exists(env_activate_fn):
        c.run(f"python3 -m venv {dev_env}")
    
    current_hash = get_requirements_hash()
    stored_hash = get_stored_hash()
    
    if current_hash != stored_hash:
        print("Requirements files have changed, installing dependencies...")
        c.run(f"source {env_activate_fn} && pip install -r ./requirements.txt")
        c.run(f"source {env_activate_fn} && pip install -r ./requirements-dev.txt")
        store_hash(current_hash)
    else:
        print("Requirements files unchanged, skipping pip install.")


@task(create_dev_env)
def run_workflow(c):

    check_cwd()

    print("Running the workflow ...")
    c.run(f"source {env_activate_fn} && python src/main.py", pty=True )


@task(create_dev_env)
def marimo_diagnose_sqllite_perf(c):

    check_cwd()

    fn ="notebooks/diagnose_sqllite_perf.py"

    print(f"Marimoing {fn} ...")
    c.run(f"source {env_activate_fn} && marimo edit {fn}", pty=True )

@task(create_dev_env)
def marimo_inspect_vicar_problem(c):

    check_cwd()

    fn ="notebooks/inspect_vicar_problem.py"

    print(f"Marimoing {fn} ...")

    with c.cd("./src"):
        c.run(f"source ../{env_activate_fn} && marimo edit ../{fn}", pty=True )





@task(create_dev_env)
def marimo_inspect_metadata(c):

    check_cwd()

    fn ="notebooks/inspect_metadata.py"

    print(f"Marimoing {fn} ...")

    with c.cd("./src"):
        c.run(f"source ../{env_activate_fn} && marimo edit ../{fn}", pty=True )


@task(create_dev_env)
def marimo_tune_hough_detection(c):

    check_cwd()

    fn ="notebooks/tune_hough_detection.py"

    print(f"Marimoing {fn} ...")

    with c.cd("./src"):
        os.environ["MARIMO_OUTPUT_MAX_BYTES"] = "10_000_000"
        c.run(f"source ../{env_activate_fn} && marimo edit ../{fn}", pty=True )



@task(create_dev_env)
def marimo_examine_circle_data(c):

    check_cwd()

    fn ="notebooks/examine_circle_data.py"

    print(f"Marimoing {fn} ...")

    with c.cd("./src"):
        os.environ["MARIMO_OUTPUT_MAX_BYTES"] = "10_000_000"
        c.run(f"source ../{env_activate_fn} && marimo edit ../{fn}", pty=True )