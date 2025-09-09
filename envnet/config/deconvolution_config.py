from dataclasses import dataclass
from .base_config import BaseConfig

@dataclass
class DeconvolutionConfig(BaseConfig):
    """Configuration parameters for the spectral deconvolution workflow."""

    # Overrides from BaseConfig
    mz_tol: float = 0.01  # More permissive for deconvolution
    min_rt: float = 1.0   # Stricter RT range for deconvolution
    max_rt: float = 30.0
    min_library_match_score: float = 0.8  # Higher threshold for deconvolution
    
    # Deconvolution-specific parameters
    isolation_tolerance: float = 0.5
    z_score_threshold: float = 2.5
    num_points: int = 3
    instrument_precision: float = 0.001
    filter_percent: float = 0.0
    min_intensity_ratio: float = 2.0
    file_key: str = 'ms2_neg'
    
    # Note: mdm_deltas is inherited from BaseConfig