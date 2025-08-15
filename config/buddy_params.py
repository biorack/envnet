"""
Configuration parameters for MS-Buddy analysis.

"""

from dataclasses import dataclass

@dataclass
class BuddyAnalysisConfig:
    """Configuration parameters for MS-Buddy analysis."""
    
    # Buddy analysis parameters
    polarity: str = 'negative'
    max_fdr: float = 0.05
    ms1_tol: float = 0.001
    ms2_tol: float = 0.002
    max_frag_reserved: int = 50
    rel_int_denoise_cutoff: float = 0.01
    n_cpu: int = 40
    ppm: bool = False
    halogen: bool = False
    parallel: bool = True
    batch_size: int = 10000
