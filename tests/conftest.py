"""Make the integration's engine importable as a top-level package (`import engine`) without Home Assistant."""
import sys
from pathlib import Path

ENGINE_PARENT = Path(__file__).resolve().parents[1] / "custom_components" / "solar_headroom"
if str(ENGINE_PARENT) not in sys.path:
    sys.path.insert(0, str(ENGINE_PARENT))
