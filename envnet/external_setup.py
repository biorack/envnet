"""Setup for blink external package."""
import sys
from pathlib import Path

def setup_external_packages():
    """Add blink to Python path if it exists."""
    envnet_root = Path(__file__).parent.parent
    blink_path = envnet_root / "external" / "blink"
    
    if blink_path.exists() and str(blink_path) not in sys.path:
        sys.path.insert(0, str(blink_path))

# Auto-setup when imported
setup_external_packages()

# Check blink availability
try:
    import blink
    BLINK_AVAILABLE = True
except ImportError:
    BLINK_AVAILABLE = False