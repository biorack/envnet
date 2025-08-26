from dataclasses import dataclass
from .base_config import BaseConfig

@dataclass
class AnnotationConfig(BaseConfig):
    """Configuration parameters for ENVnet annotation workflows."""

    # Overrides from BaseConfig
    mz_tol: float = 0.01   # More permissive for annotation
    min_score: float = 0.5  # Lower threshold for annotation flexibility

    # Annotation-specific parameters
    min_ms1_datapoints: int = 5  # minimum number of MS1 data points for a feature to be considered
    chunk_size: int = 1000       # chunk size for MS2 scoring
    ppm_tolerance: float = 5.0   # ppm tolerance for node atlas creation    
    # Note: min_matches, override_matches, intensity_power, bin_width, 
    # min_rt, max_rt, metadata_folder, module_path all inherited from BaseConfig