"""
Chemical formula analysis tools for ENVnet.

Contains functions for parsing chemical formulas and calculating
molecular properties like aromaticity indices, oxidation states, etc.

This module is now completely self-contained! You can import and use these functions in your other modules like:

from .formula_tools import get_formula_props, calculate_mass, formula_to_dict

"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
from rdkit.Chem import rdchem


def formula_to_dict(formula: str) -> Dict[str, int]:
    """
    Convert chemical formula string to dictionary of element counts.
    
    Args:
        formula: Chemical formula string (e.g., 'C6H12O6')
        
    Returns:
        Dictionary mapping element symbols to counts (lowercase keys)
        
    Example:
        >>> formula_to_dict('C6H12O6')
        {'c': 6, 'h': 12, 'o': 6}
    """
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    element_dict = {k.lower(): v for k, v in matches}
    
    for k, v in element_dict.items():
        if v == '':
            element_dict[k] = 1
        else:
            element_dict[k] = int(v)
            
    return element_dict


def calculate_mass(formula: str) -> float:
    """
    Calculate molecular mass from chemical formula.
    
    Args:
        formula: Chemical formula string
        
    Returns:
        Molecular mass in daltons
    """
    pattern = r'([A-Z][a-z]*)(\d*)'    
    mass = 0
    pt = rdchem.GetPeriodicTable()

    for el, count in re.findall(pattern, formula):
        count = int(count) if count else 1
        mass += pt.GetMostCommonIsotopeMass(el) * count
        
    return mass


def calc_dbe(element_counts: Dict[str, int]) -> float:
    """
    Calculate double bond equivalents (DBE).
    
    Reference:
    D'Andrilli, J., et al. (2010). Comprehensive characterization of marine 
    dissolved organic matter by Fourier transform ion cyclotron resonance mass 
    spectrometry with electrospray and atmospheric pressure photoionization. 
    Rapid Communications in Mass Spectrometry, 24(5), 643–650.
    
    Args:
        element_counts: Dictionary of element counts (lowercase keys)
        
    Returns:
        Double bond equivalents
    """
    m = element_counts
    dbe = 1 + m.get('c', 0) - m.get('h', 0)/2 + m.get('n', 0)/2
    return dbe


def calc_dbe_ai(element_counts: Dict[str, int]) -> float:
    """
    Calculate DBE for aromaticity index calculation.
    
    Reference:
    Koch, B. P. and Dittmar, T. (2006). From mass to structure: an aromaticity 
    index for high-resolution mass data of natural organic matter. 
    Rapid Commun. Mass Spectrom. 20, 926–932.
    
    Args:
        element_counts: Dictionary of element counts (lowercase keys)
        
    Returns:
        DBE for aromaticity index
    """
    m = element_counts
    dbe_ai = (1 + m.get('c', 0) - m.get('o', 0) - m.get('s', 0) - 
              m.get('h', 0)/2 - m.get('n', 0)/2 - m.get('p', 0)/2)
    return dbe_ai


def calc_dbe_ai_mod(element_counts: Dict[str, int]) -> float:
    """
    Calculate modified DBE for aromaticity index calculation.
    
    Reference:
    Koch, B. P. and Dittmar, T. (2006). From mass to structure: an aromaticity 
    index for high-resolution mass data of natural organic matter. 
    Rapid Commun. Mass Spectrom. 20, 926–932.
    
    Args:
        element_counts: Dictionary of element counts (lowercase keys)
        
    Returns:
        Modified DBE for aromaticity index
    """
    m = element_counts
    dbe_ai_mod = (1 + m.get('c', 0) - m.get('o', 0)/2 - m.get('s', 0) - 
                  m.get('h', 0)/2 - m.get('n', 0)/2 - m.get('p', 0)/2)
    return dbe_ai_mod


def aromaticity_index(element_counts: Dict[str, int]) -> Optional[float]:
    """
    Calculate aromaticity index (AI).
    
    Reference:
    Koch, B. P. and Dittmar, T. (2006). From mass to structure: an aromaticity 
    index for high-resolution mass data of natural organic matter. 
    Rapid Commun. Mass Spectrom. 20, 926–932.
    
    Args:
        element_counts: Dictionary of element counts (lowercase keys)
        
    Returns:
        Aromaticity index or None if undefined
    """
    m = element_counts
    dbe_ai = calc_dbe_ai(m)
    denominator = m.get('c', 0) - m.get('o', 0) - m.get('n', 0) - m.get('s', 0) - m.get('p', 0)
    
    if denominator == 0:
        return None
        
    ai = dbe_ai / denominator
    return ai


def modified_aromaticity_index(element_counts: Dict[str, int]) -> Optional[float]:
    """
    Calculate modified aromaticity index (AI_mod).
    
    Reference:
    Koch, B. P. and Dittmar, T. (2006). From mass to structure: an aromaticity 
    index for high-resolution mass data of natural organic matter. 
    Rapid Commun. Mass Spectrom. 20, 926–932.
    
    Args:
        element_counts: Dictionary of element counts (lowercase keys)
        
    Returns:
        Modified aromaticity index or None if undefined
    """
    m = element_counts
    dbe_ai_mod = calc_dbe_ai_mod(m)
    denominator = m.get('c', 0) - m.get('o', 0)/2 - m.get('n', 0) - m.get('s', 0) - m.get('p', 0)
    
    if denominator == 0:
        return None
        
    ai_mod = dbe_ai_mod / denominator
    return ai_mod


def calc_nosc(element_counts: Dict[str, int]) -> float:
    """
    Calculate nominal oxidation state of carbon (NOSC).
    
    Reference:
    Riedel, T., Biester, H., and Dittmar, T. (2012). Molecular Fractionation 
    of Dissolved Organic Matter with Metal Salts. Environmental Science & 
    Technology 46 (8), 4419-4426.
    
    Args:
        element_counts: Dictionary of element counts (lowercase keys)
        
    Returns:
        Nominal oxidation state of carbon
    """
    m = element_counts
    c_count = m.get('c', 0)
    
    if c_count > 0:
        nosc = 4 - (4*c_count + m.get('h', 0) - 3*m.get('n', 0) - 
                   2*m.get('o', 0) - 2*m.get('s', 0)) / c_count
    else:
        nosc = np.inf
        
    return nosc


def run_formula_calcs(element_counts: Dict[str, int]) -> Optional[Dict[str, Union[float, int, None]]]:
    """
    Run all formula calculations for a given element composition.
    
    Args:
        element_counts: Dictionary of element counts (lowercase keys)
        
    Returns:
        Dictionary containing all calculated properties, or None if no carbon
    """
    m = element_counts
    
    # Ensure all elements are present
    elements = ['c', 'h', 'n', 'o', 's', 'p']
    for e in elements:
        if e not in m:
            m[e] = 0
    
    if m['c'] == 0:
        return None
    
    # Calculate N:P ratio
    if m['p'] == 0:
        n_to_p = None
    else:
        n_to_p = m['n'] / m['p']
    
    results = {
        'dbe': calc_dbe(m),
        'dbe_ai': calc_dbe_ai(m),
        'dbe_ai_mod': calc_dbe_ai_mod(m),
        'ai_mod': modified_aromaticity_index(m),
        'ai': aromaticity_index(m),
        'nosc': calc_nosc(m),
        'h_to_c': m['h'] / m['c'],
        'o_to_c': m['o'] / m['c'],
        'n_to_c': m['n'] / m['c'],
        'p_to_c': m['p'] / m['c'],
        'n_to_p': n_to_p,
        'c': m['c'],
        'h': m['h'],
        'o': m['o'],
        'n': m['n'],
        's': m['s'],
        'p': m['p']
    }
    
    return results


def get_formula_props(df: pd.DataFrame, formula_key: str = 'formula_rank_1') -> pd.DataFrame:
    """
    Calculate formula properties for all unique formulas in a DataFrame.
    
    Args:
        df: DataFrame containing chemical formulas
        formula_key: Column name containing the formulas
        
    Returns:
        DataFrame with formula properties for each unique formula
    """
    formula_props = []
    
    # Get unique formulas that contain carbon
    my_formulas = df.loc[pd.notna(df[formula_key]), formula_key].unique()
    my_formulas = [f for f in my_formulas if 'C' in f]
    
    for formula in my_formulas:
        element_dict = formula_to_dict(formula)
        props = run_formula_calcs(element_dict)
        
        if props is not None:
            props['formula'] = formula
            formula_props.append(props)
    
    return pd.DataFrame(formula_props)