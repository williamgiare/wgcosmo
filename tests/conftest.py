import importlib.util
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wgcosmo-matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "statistics/MCMC_Evidence"))


def load_module(relative):
    path = ROOT / relative
    name = "wgcosmo_test_" + "_".join(path.relative_to(ROOT).with_suffix("").parts)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
