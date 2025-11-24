import sys
import pathlib
import types
import pytest

# Ensure src is importable
repo_root = pathlib.Path(__file__).parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Ensure log directory exists for database logging handlers
logs_path = repo_root / "logs"
logs_path.mkdir(exist_ok=True)


def _install_stubs(target_dict):
    """Install lightweight stubs for external modules."""
    pvl_stub = types.SimpleNamespace()
    collections = types.SimpleNamespace()

    class PVLObject(list):
        pass

    class PVLModule(list):
        pass

    class Quantity:
        def __init__(self, value, units):
            self.value = value
            self.units = units

    collections.PVLObject = PVLObject
    collections.PVLModule = PVLModule
    collections.Quantity = Quantity

    def loads(text: str):
        module = PVLModule()
        for line in text.strip().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                module.append((key.strip(), value.strip()))
        return module

    pvl_stub.collections = collections
    pvl_stub.loads = loads
    target_dict["pvl"] = pvl_stub

    class DummyVicarImage:
        def __init__(self, path):
            self.path = path
            self.array = []

    target_dict["vicar"] = types.SimpleNamespace(VicarImage=DummyVicarImage)
    target_dict["cv2"] = types.SimpleNamespace(HOUGH_GRADIENT=1, HOUGH_GRADIENT_ALT=2)


# Install stubs eagerly for import-time usage
_install_stubs(sys.modules)


@pytest.fixture(autouse=True)
def _reset_stubs(monkeypatch):
    modules = {"pvl": sys.modules["pvl"], "vicar": sys.modules["vicar"], "cv2": sys.modules["cv2"]}
    yield
    _install_stubs(sys.modules)


@pytest.fixture()
def in_memory_sessionmaker():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import database

    engine = create_engine("sqlite:///:memory:")
    database.Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
