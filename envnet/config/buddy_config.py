from dataclasses import dataclass
from .base_config import BaseConfig

@dataclass
class BuddyConfig(BaseConfig):
    """Configuration parameters for MS-Buddy analysis."""
    
    # MS-Buddy specific tolerance parameters
    ms1_tol: float = 0.001  # Separate from base mz_tol for MS1 specificity
    ms2_tol: float = 0.002  # Separate from base mz_tol for MS2 specificity
    
    # MS-Buddy analysis parameters
    max_fdr: float = 0.05
    max_frag_reserved: int = 50
    rel_int_denoise_cutoff: float = 0.01
    n_cpu: int = 40
    ppm: bool = False
    halogen: bool = False
    parallel: bool = True
    batch_size: int = 10000
    
    # Note: polarity inherited from BaseConfig