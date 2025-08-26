from dataclasses import dataclass
from .base_config import BaseConfig

@dataclass
class AnalysisConfig(BaseConfig):
    """Configuration parameters for ENVnet analysis workflows."""

    # Overrides from BaseConfig - more permissive for analysis
    mz_tol: float = 0.01   
    min_score: float = 0.5  
    
    # Analysis-specific parameters
    min_ms1_datapoints: int = 5
    require_ms2_support: bool = False  # Whether to require MS2 support for MS1 features
    
    # Statistical analysis parameters
    max_pvalue: float = 0.05
    normalize_data: bool = True
    peak_value: str = 'peak_area'  # 'peak_area' or 'peak_height'
    
    # PCA parameters
    n_components: int = 2
    log_transform: bool = True
    
    # Visualization parameters
    min_upset_subset_size: int = 100
    figure_dpi: int = 300
    font_size: int = 14
    
    # File paths for annotation results (can be overridden)
    ms1_results_file: str = None
    ms2_deconv_results_file: str = None  
    ms2_original_results_file: str = None