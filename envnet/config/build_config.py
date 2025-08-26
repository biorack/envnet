from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np
from .base_config import BaseConfig

@dataclass
class BuildConfig(BaseConfig):
    """Configuration parameters for ENVnet construction."""
    
    # Build-specific parameters
    # deduplication
    min_cluster_size: int = 2

    # networking
    remblink_cutoff: float = 0.05
    min_observations: int = 2
    network_max_mz_difference: float = 1000.0
    # Processing
    chunk_size: int = 2000
    verbose: bool = True

    # Reference data (populated in __post_init__)
    ref_spec: Optional[List] = None
    ref_spec_nl: Optional[List] = None
    ref_pmz: Optional[List] = None
    ref_pmz_nl: Optional[List] = None
    ref: Optional[pd.DataFrame] = None
    ref2: Optional[pd.DataFrame] = None
    
    # Note: mdm_df and mdm_masses are inherited from BaseConfig
    
    def __post_init__(self):
        """Initialize derived parameters after BaseConfig loads MDM data."""
        # Call parent __post_init__ to load MDM data
        super().__post_init__()
        
        # Load reference data using inherited mdm_df
        from ..build.reference import load_p2d2_reference_data
        ref, ref2 = load_p2d2_reference_data(self.mdm_df, mz_tol=self.mz_tol)
        
        # Clean up reference data
        bad = []
        for i, row in ref.iterrows():
            num_ions = row['spectrum'].shape[1]
            if num_ions == 0:
                bad.append(i)
        ref.drop(index=bad, inplace=True)
        ref.reset_index(inplace=True, drop=True)
        
        # Populate reference attributes
        self.ref = ref
        self.ref2 = ref2
        self.ref_spec = ref['spectrum'].tolist()
        self.ref_pmz = ref['precursor_mz'].tolist()
        self.ref_spec_nl = ref2['nl_spectrum'].tolist()
        self.ref_pmz_nl = ref2['precursor_mz'].tolist()
