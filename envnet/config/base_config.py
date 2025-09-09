from dataclasses import dataclass, fields
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, List
import os
import yaml


def find_envnet_root() -> Path:
    """Find the envnet root directory."""
    return Path(__file__).resolve().parent.parent.parent

ENVNET_ROOT = find_envnet_root()

@dataclass
class BaseConfig:
    """Base configuration with common parameters."""
    file_metadata_source: str = 'local_csv'
    file_metadata_path: Optional[str] = None
    # Core tolerance parameters
    mz_tol: float = 0.002
    
    # RT parameters  
    min_rt: float = 0.5
    max_rt: float = 70.0
    
    # Scoring parameters
    min_matches: int = 3
    override_matches: int = 20
    intensity_power: float = 0.5
    bin_width: float = 0.001
    min_deduplication_score: float = 0.9  # High score for finding identical spectra
    min_library_match_score: float = 0.7  # Lower score for matching against a library    

    # File paths
    metadata_folder: str = '/global/cfs/cdirs/metatlas/projects/carbon_network'
    module_path: str = str(ENVNET_ROOT)
    model_file: str = os.path.join(module_path, 'envnet', 'data', 'mdm_negative_random_forest.joblib')
    # Instrument parameters
    polarity: str = 'negative'
    
    # MDM (Mass Defect Matching) data - loaded once, used by multiple configs
    mdm_df: Optional[pd.DataFrame] = None
    mdm_deltas: Optional[Dict[str, float]] = None  # dict format for deconvolution
    mdm_masses: Optional[List[float]] = None       # list format for build
    
    @classmethod
    def from_file(cls, file_path: str):
        """
        Creates a config instance by loading parameters from a YAML file.
        Any parameters in the YAML file will override the class defaults.
        """
        # If no file is provided, return a default config instance
        if not file_path:
            return cls()

        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}

        # Get the set of valid field names for this dataclass
        valid_fields = {f.name for f in fields(cls)}
        
        # Filter the loaded data to only include keys that are valid fields
        # in this dataclass. This prevents errors if the YAML file has extra keys.
        filtered_data = {k: v for k, v in config_data.items() if k in valid_fields}
        
        # Create an instance of the class, where values from the file
        # override the defaults.
        return cls(**filtered_data)
    
    def __post_init__(self):
        """Load MDM data after object creation."""
        mdm_path = os.path.join(self.module_path, 'envnet','data', 'mdm_neutral_losses.csv')
        if os.path.exists(mdm_path):
            self.mdm_df = pd.read_csv(mdm_path)
            self.mdm_deltas = self.mdm_df.set_index('difference')['mass'].to_dict()
            self.mdm_masses = [0] + self.mdm_df['mass'].tolist()
        else:
            raise FileNotFoundError(f"MDM neutral losses file not found at {mdm_path}")