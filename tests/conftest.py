import sys
from pathlib import Path

# Add project root and src directory to sys.path
proj_root = Path(__file__).resolve().parent.parent
src_dir = proj_root / "src"
for p in [str(proj_root), str(src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

collect_ignore = []

# Prevent pytest from hanging on Windows machines or environments without native Triton/XLA compiler wheels
if sys.platform == "win32":
    collect_ignore.extend(["test_ultrametric.py", "test_ultrametric_jax.py"])

try:
    import jax
except ImportError:
    if "test_ultrametric_jax.py" not in collect_ignore:
        collect_ignore.append("test_ultrametric_jax.py")

