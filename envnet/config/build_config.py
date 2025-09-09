from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np
from .base_config import BaseConfig
import os
@dataclass
class BuildConfig(BaseConfig):
    """Configuration parameters for ENVnet construction."""
    
    # Build-specific parameters
    # deduplication
    min_cluster_size: int = 4
    
    # Override Base Config. High threshold for network building
    # the higher threshold seems to minimize bridging artifacts
    # networking
    remblink_cutoff: float = 0.05
    # min_observations: int = 5 #this is never used.
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
        """Initialize derived parameters and load file metadata."""
        super().__post_init__()
        
        # --- Load file metadata based on the specified source ---
        if self.file_metadata_source == 'local_csv':
            if self.file_metadata_path and os.path.exists(self.file_metadata_path):
                print(f"Loading file list from CSV: {self.file_metadata_path}")
                self.file_list_df = pd.read_csv(self.file_metadata_path)
            elif self.file_metadata_path:
                raise FileNotFoundError(f"Specified CSV file not found: {self.file_metadata_path}")
            else:
                raise ValueError("file_metadata_path must be specified for local_csv source.")
        
        elif self.file_metadata_source == 'google_sheets':
            print("Loading file list from Google Sheets...")
            from ..utils import get_google_sheet  # Local import to avoid circular dependencies
            
            # Load file data
            file_df = get_google_sheet(notebook_name='Supplementary Tables', sheet_name='Table 1a')
            file_df.columns = file_df.iloc[0]
            file_df = file_df[1:].reset_index(drop=True)
            
            # Load environmental class info
            envo_name = get_google_sheet(notebook_name='Supplementary Tables', sheet_name='Table 1b')
            envo_name.columns = envo_name.iloc[0]
            envo_name = envo_name[1:].reset_index(drop=True)
            
            # Merge and clean
            file_df = pd.merge(file_df, envo_name[['name', 'id', 'common parent name']], 
                             left_on='environmental_subclass', right_on='id', how='inner')
            
            self.file_list_df = file_df
        
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
