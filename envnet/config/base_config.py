from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, List
import os

def find_envnet_root() -> Path:
    """Find the envnet root directory."""
    return Path(__file__).resolve().parent.parent.parent

ENVNET_ROOT = find_envnet_root()

@dataclass
class BaseConfig:
    """Base configuration with common parameters."""
    
    # Core tolerance parameters
    mz_tol: float = 0.002
    
    # RT parameters  
    min_rt: float = 0.5
    max_rt: float = 60.0
    
    # Scoring parameters
    min_matches: int = 3
    override_matches: int = 20
    intensity_power: float = 0.5
    bin_width: float = 0.001
    min_score: float = 0.7
    
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
    
    def __post_init__(self):
        """Load MDM data after object creation."""
        mdm_path = os.path.join(self.module_path, 'envnet','data', 'mdm_neutral_losses.csv')
        if os.path.exists(mdm_path):
            self.mdm_df = pd.read_csv(mdm_path)
            self.mdm_deltas = self.mdm_df.set_index('difference')['mass'].to_dict()
            self.mdm_masses = [0] + self.mdm_df['mass'].tolist()
        else:
            raise FileNotFoundError(f"MDM neutral losses file not found at {mdm_path}")