import pathlib

def extract_stem(filename):

    path = pathlib.Path(filename)
    # Keep removing the last suffix until none are left
    while path.suffix:
        path = path.with_suffix('')

    return path.name
