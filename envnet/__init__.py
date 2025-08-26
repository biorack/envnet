"""
ENVnet: Environmental Metabolomics Networking Tools
"""

__version__ = "0.1.0"

# Set up external packages (blink)
from . import external_setup
from . import deconvolution
from . import build  
from . import annotation
# from . import analysis

from . import config

# Make vendored functions easily accessible
from .vendor import get_google_sheet, ft

# Configuration imports
from .config import BaseConfig, DeconvolutionConfig

# Build module imports (new)
try:
    from . import build
    from .build import ENVnetBuilder, build_envnet, quick_envnet
    _build_available = True
except ImportError:
    _build_available = False

__all__ = [
    # Existing exports
    'BaseConfig',
    'DeconvolutionConfig', 
    'get_google_sheet',
    'ft',
    "deconvolution", "build", "annotation", "analysis", "config"
]

# Add build exports if available
if _build_available:
    __all__.extend([
        'build',
        'ENVnetBuilder',
        'build_envnet', 
        'quick_envnet',
    ])



__version__ = "0.1.0"
