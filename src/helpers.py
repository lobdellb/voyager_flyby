import pathlib
import re
import os

def extract_stem(filename):

    path = pathlib.Path(filename)
    # Keep removing the last suffix until none are left
    while path.suffix:
        path = path.with_suffix('')

    return path.name

def extract_prefix_from_filename(file_path):
    """
    Extracts the part of the filename before the first underscore ('_').

    Args:
        file_path (str): The full path to the file.

    Returns:
        str: The part of the filename before the first underscore.
    """

    # Get the filename from the path
    filename = os.path.basename(file_path)

    # Use regex to extract the part before the first underscore
    match = re.match(r"([^_]+)_", filename)
    if match:
        return match.group(1)
    return None
