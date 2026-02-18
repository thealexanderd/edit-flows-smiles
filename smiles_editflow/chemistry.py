"""Chemistry utilities using RDKit for SMILES validation and manipulation."""

from typing import Optional
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors

# Suppress RDKit parser logs for invalid/corrupt SMILES seen during data filtering.
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Convert SMILES string to RDKit Mol object.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Mol object or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def is_valid_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string is valid.
    
    Args:
        smiles: SMILES string
        
    Returns:
        True if valid, False otherwise
    """
    mol = smiles_to_mol(smiles)
    return mol is not None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize a SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Canonical SMILES or None if invalid
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def largest_fragment(smiles: str) -> Optional[str]:
    """
    Extract the largest fragment from a SMILES string (strips salts/mixtures).
    
    Args:
        smiles: SMILES string
        
    Returns:
        SMILES of largest fragment or None if invalid
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    
    # Split on '.' and get largest fragment by number of atoms
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) == 0:
        return None
    
    # Get fragment with most atoms
    largest = max(frags, key=lambda m: m.GetNumAtoms())
    
    try:
        return Chem.MolToSmiles(largest)
    except Exception:
        return None


def randomized_smiles(smiles: str, seed: Optional[int] = None) -> Optional[str]:
    """
    Generate a randomized SMILES string (different atom ordering).
    
    Uses RDKit's doRandom flag for MolToSmiles.
    
    Args:
        smiles: SMILES string
        seed: Random seed for reproducibility (not directly supported by RDKit)
        
    Returns:
        Randomized SMILES or None if invalid
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    
    try:
        # RDKit's MolToSmiles with doRandom=True
        return Chem.MolToSmiles(mol, doRandom=True)
    except Exception:
        return None


def get_num_atoms(smiles: str) -> Optional[int]:
    """
    Get the number of atoms in a molecule.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Number of atoms or None if invalid
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return mol.GetNumAtoms()


def filter_smiles(smiles: str, max_len: int = 100, allow_disconnected: bool = False) -> bool:
    """
    Filter SMILES based on criteria.
    
    Args:
        smiles: SMILES string
        max_len: Maximum number of tokens allowed
        allow_disconnected: Whether to allow disconnected molecules (with '.')
        
    Returns:
        True if molecule passes filters, False otherwise
    """
    # Check validity
    if not is_valid_smiles(smiles):
        return False
    
    # Check disconnected
    if not allow_disconnected and '.' in smiles:
        return False
    
    # Check length
    from .tokenizer import tokenize
    tokens = tokenize(smiles)
    if len(tokens) > max_len:
        return False
    
    return True


if __name__ == "__main__":
    # Test the chemistry utilities
    test_smiles = [
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "invalid",  # Invalid
        "CC(=O)O.Cl",  # With salt
    ]
    
    print("Chemistry utilities test:")
    for smi in test_smiles:
        print(f"\nTesting: {smi}")
        print(f"  Valid: {is_valid_smiles(smi)}")
        print(f"  Canonical: {canonicalize_smiles(smi)}")
        print(f"  Largest fragment: {largest_fragment(smi)}")
        print(f"  Randomized: {randomized_smiles(smi)}")
        print(f"  Num atoms: {get_num_atoms(smi)}")
