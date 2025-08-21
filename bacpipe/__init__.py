import logging
import zipfile
from pathlib import Path
import importlib.resources as pkg_resources

# Unzip model_checkpoints.zip to initiate models dir structure (package-relative)
try:
    with pkg_resources.path(__package__, "model_checkpoints.zip") as zip_path:
        checkpoints_dir = Path(__file__).parent / "model_checkpoints"
        if not checkpoints_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(checkpoints_dir)
except Exception as e:
    pass

# Initialize Logger
logger = logging.getLogger("bacpipe")
c_handler = logging.StreamHandler()
c_format = logging.Formatter("%(name)s::%(levelname)s:%(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
logger.setLevel(logging.WARNING)
