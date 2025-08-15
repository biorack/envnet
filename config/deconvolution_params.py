import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple  # Add Tuple import
from pathlib import Path

@dataclass
class DeconvolutionConfig:
    """Configuration parameters for the spectral deconvolution workflow."""

    # Tolerance parameters
    mz_tolerance: float = 0.01
    isolation_tolerance: float = 0.5
    
    # Clustering parameters
    z_score_threshold: float = 2.5
    num_points: int = 3
    instrument_precision: float = 0.001
    
    # RT filtering
    min_rt: float = 1.0
    max_rt: float = 7.0
    
    # Intensity filtering
    filter_percent: float = 0.0
    min_intensity_ratio: float = 2.0
    similarity_cutoff: float = 0.8
    
    # File parameters
    file_key: str = 'ms2_neg'
    
    # MDM deltas - either provide the dict directly or a path to load from
    mdm_deltas: Optional[Dict[str, float]] = None
    mdm_deltas_path: Optional[str] = None
    
    # Module/project root path
    module_path: Optional[str] = None
    
    def __post_init__(self):
        """Load MDM deltas if not provided and set module path."""
        if self.mdm_deltas is None:
            if self.mdm_deltas_path is None:
                # Find both the MDM file and module path
                self.mdm_deltas_path, self.module_path = self._find_paths()
            
            if self.mdm_deltas_path and Path(self.mdm_deltas_path).exists():
                self.mdm_deltas = pd.read_csv(self.mdm_deltas_path).set_index('difference')['mass'].to_dict()
            else:
                raise FileNotFoundError(f"MDM deltas file not found at {self.mdm_deltas_path}")
    
    def _find_paths(self) -> Tuple[Optional[str], Optional[str]]:
        """Find the MDM deltas file and module path by walking up the directory tree.
        
        Returns:
            Tuple of (mdm_deltas_path, module_path)
        """
        from pathlib import Path
        
        current_path = Path(__file__).parent
        
        # Walk up the directory tree to find envnet root
        for parent in [current_path] + list(current_path.parents):
            potential_mdm_file = parent / 'data' / 'mdm_neutral_losses.csv'
            if potential_mdm_file.exists():
                return str(potential_mdm_file), str(parent)
        
        # Fallback: assume a standard structure
        envnet_root = current_path.parents[1] if len(current_path.parents) > 1 else current_path.parent
        fallback_mdm_path = envnet_root / 'data' / 'mdm_neutral_losses.csv'
        
        mdm_path = str(fallback_mdm_path) if fallback_mdm_path.exists() else None
        module_path = str(envnet_root)
        
        return mdm_path, module_path