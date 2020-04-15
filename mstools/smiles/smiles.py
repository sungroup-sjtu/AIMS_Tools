#!/usr/bin/env python3
# # coding=utf-8
import pybel
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
import os
CWD = os.path.dirname(os.path.abspath(__file__))
from .fingerprint import *


def get_para(T=298, eps=None, sigma=None, lamb=None):
    f = 1 + lamb * 0.01 * (T - 298)
    eps = eps * f ** 2 * 4.184
    sigma = (2 * f) ** (-1 / 6) * sigma
    return eps, sigma


def get_para_298(T=298, eps=None, sigma=None, lamb=None):
    f = 1 + lamb * 0.01 * (T - 298)
    eps /= (f ** 2 * 4.184)
    sigma /= ((2 * f) ** (-1 / 6))
    return sigma / 2 ** (1 / 6), eps * 16 / 27


def add_atom_index(mol):
    atoms = mol.GetAtoms()
    for atom in atoms:
        atom.SetProp('molAtomMapNumber', str(atom.GetIdx()))
    return mol


# get basic information of molecules
def get_canonical_smiles(smiles):
    rdk_mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(rdk_mol)


def get_rdkit_smiles(smiles):
    rdk_mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(rdk_mol)


def get_atom_numbers(smiles):
    mol = pybel.readstring("smi", smiles)
    mol.addh()
    return len(mol.atoms)


def get_heavy_atom_numbers(smiles):
    py_mol = pybel.readstring("smi", smiles)
    py_mol.removeh()
    return len(py_mol.atoms)


def get_AtomNum_list(smiles):
    rdk_mol = Chem.MolFromSmiles(smiles)
    atom_type_list = []
    for atom in rdk_mol.GetAtoms():
        if atom.GetAtomicNum() not in atom_type_list:
            atom_type_list.append(atom.GetAtomicNum())
    if 1 in atom_type_list:
        atom_type_list.remove(1)
    atom_type_list.sort()
    return atom_type_list


def get_charge(smiles):
    mol = pybel.readstring("smi", smiles)
    return mol.charge


def get_ring_number(smiles):
    rdk_mol = Chem.MolFromSmiles(smiles)
    return rdk_mol.GetRingInfo().NumRings()


# stereo functions
def has_stereo_isomer(smiles):
    if len(get_stereo_isomer(smiles)) == 1:
        return False
    else:
        return True


def get_stereo_isomer(smiles, canonical=False): # There is a bug for canonical=True, rarely happen.
    rdk_mol = Chem.MolFromSmiles(smiles)
    isomers = tuple(EnumerateStereoisomers(rdk_mol))
    smiles_list = []
    for smi in sorted(Chem.MolToSmiles(x, isomericSmiles=True) for x in isomers):
        if canonical:
            smi = get_canonical_smiles(smi)
        smiles_list.append(smi)
    return smiles_list


def has_stereo_isomer_from_inchi(inchi):
    from subprocess import Popen, PIPE
    cmd = 'python3 %s/../bin/IsExplicitInchi.py -i %s' % (CWD, inchi)
    warning = str(Popen(cmd.split(), stdout=None, stderr=PIPE).communicate()[1])
    if 'Omitted undefined stereo' in warning:
        return True
    else:
        return False


def remove_chirality(smiles):
    s = ''.join(smiles.split('@'))
    s = ''.join(s.split('H'))
    s = ''.join(s.split('['))
    s = ''.join(s.split(']'))
    s = ''.join(s.split('/'))
    s = ''.join(s.split('\\'))
    return get_canonical_smiles(s)


# classification functions
def is_mol_stable(smiles):
    rdk_mol = Chem.MolFromSmiles(smiles)
    # double bond check
    atom_idx_list = []
    for bond in rdk_mol.GetBonds():
        if bond.GetBondTypeAsDouble() == 2:
            atom_idx_list.append(bond.GetBeginAtomIdx())
            atom_idx_list.append(bond.GetEndAtomIdx())
    if len(set(atom_idx_list)) != len(atom_idx_list):
        return False
    # two hydroxyl group cannot connect to same carbon

    return True


def is_aromatic(smiles):
    rdk_mol = Chem.MolFromSmiles(smiles)
    if rdk_mol.GetAromaticAtoms() == 0:
        return False
    else:
        return True


def selection(smiles, type=None):
    if type is None:
        return False
    rdk_mol = Chem.MolFromSmiles(smiles)
    if rdk_mol is None:
        return False
    if type == 'CH':
        for atom in rdk_mol.GetAtoms():
            if atom.GetAtomicNum() not in [1, 6]:
                return False
    elif type == 'LinearBranchAlkane':
        for atom in rdk_mol.GetAtoms():
            if atom.GetAtomicNum() not in [1, 6]:
                return False
        for bond in rdk_mol.GetBonds():
            if bond.GetBondTypeAsDouble() != 1:
                return False
        if rdk_mol.GetRingInfo().NumRings() != 0:
            return False
    return True


# similarity functions
def similarity_comparison(smiles1, smiles2, useChirality=False):
    from rdkit.Chem import AllChem as Chem
    from rdkit import DataStructs
    rdk_mol1 = Chem.MolFromSmiles(smiles1)
    fp1 = Chem.GetMorganFingerprintAsBitVect(rdk_mol1, 2, useChirality=useChirality)
    rdk_mol2 = Chem.MolFromSmiles(smiles2)
    fp2 = Chem.GetMorganFingerprintAsBitVect(rdk_mol2, 2, useChirality=useChirality)
    # print(smiles1, smiles2)
    return DataStructs.DiceSimilarity(fp1, fp2)


def rdk_similarity_comparison(smiles1, smiles2, maxPath=7):
    from rdkit.Chem import AllChem as Chem
    from rdkit import DataStructs
    rdk_mol1 = Chem.MolFromSmiles(smiles1)
    fp1 = Chem.RDKFingerprint(rdk_mol1, maxPath=maxPath)
    rdk_mol2 = Chem.MolFromSmiles(smiles2)
    fp2 = Chem.RDKFingerprint(rdk_mol2, maxPath=maxPath)
    # print(smiles1, smiles2)
    return DataStructs.FingerprintSimilarity(fp1, fp2, metric=DataStructs.DiceSimilarity)


def get_similarity_score(smiles, smiles_list, cutoff=None):
    score = 0.
    for s in smiles_list:
        score += similarity_score(smiles, s)
        if cutoff != None and score > cutoff:
            return score
    return score


def is_similar(smiles, smiles_list, cutoff):
    score = 0.
    for s in smiles_list:
        score += similarity_score(smiles, s)
        if score > cutoff:
            return True
    return False


def similarity_score(smiles1, smiles2, type='morgan'):  # a switch function of similarity_comparison
    if type == 'rdk':
        a = rdk_similarity_comparison(smiles1, smiles2)
    else:
        a = similarity_comparison(smiles1, smiles2)
    cut = 0.6
    if a < cut:
        return 0.
    else:
        return (a - cut) / (1 - cut)
