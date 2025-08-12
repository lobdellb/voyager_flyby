import pytest
import sys
import pathlib

# Add src to path so we can import helpers
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from helpers import extract_stem


class TestExtractStem:
    """Test the extract_stem function."""
    
    def test_single_extension(self):
        """Test file with single extension."""
        assert extract_stem("file.txt") == "file"
        assert extract_stem("image.jpg") == "image"
        assert extract_stem("data.csv") == "data"
    
    def test_multiple_extensions(self):
        """Test file with multiple extensions."""
        assert extract_stem("archive.tar.gz") == "archive"
        assert extract_stem("backup.tar.bz2") == "backup"
        assert extract_stem("file.txt.backup") == "file"
    
    def test_no_extension(self):
        """Test file without extension."""
        assert extract_stem("filename") == "filename"
        assert extract_stem("README") == "README"
    
    def test_hidden_files(self):
        """Test hidden files (starting with dot)."""
        assert extract_stem(".hidden") == ".hidden"
        assert extract_stem(".gitignore") == ".gitignore"
        assert extract_stem(".hidden.txt") == ".hidden"
    
    def test_with_path(self):
        """Test file with directory path."""
        assert extract_stem("/path/to/file.txt") == "file"
        assert extract_stem("../data/archive.tar.gz") == "archive"
        assert extract_stem("folder/subfolder/image.jpg") == "image"
    
    def test_complex_extensions(self):
        """Test files with many extensions."""
        assert extract_stem("file.backup.old.txt") == "file"
        assert extract_stem("data.processed.clean.csv") == "data"
    
    def test_edge_cases(self):
        """Test edge cases."""
        assert extract_stem("") == ""
        assert extract_stem(".") == ""
        assert extract_stem("..") == ".."
        assert extract_stem("file.") == "file."
    
    def test_pathlib_path_object(self):
        """Test that function works with pathlib.Path objects."""
        path_obj = pathlib.Path("file.txt")
        assert extract_stem(path_obj) == "file"
        
        path_obj = pathlib.Path("archive.tar.gz")
        assert extract_stem(path_obj) == "archive"
    
    def test_vicar_filenames(self):
        """Test with VICAR-style filenames relevant to the project."""
        assert extract_stem("C1234567_GEOMED.IMG") == "C1234567_GEOMED"
        assert extract_stem("C1234567_GEOMED.LBL") == "C1234567_GEOMED"
        assert extract_stem("VGISS_6101.tar.gz") == "VGISS_6101"