import math
import os
import random

from .errors import OpenBabelError
import subprocess


def greatest_common_divisor(numbers):
    '''
    calculate the greatest common divisor
    '''
    minimal = min(numbers)
    for i in range(minimal, 1, -1):
        flag = True
        for number in numbers:
            if number % i != 0:
                flag = False
        if flag:
            return i
    return 1


def random_string(length=8):
    import random, string
    return ''.join(random.sample(string.ascii_letters, length))


def cd_or_create_and_cd(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            raise Exception('Cannot create directory: %s' % dir)

    try:
        os.chdir(dir)
    except:
        raise Exception('Cannot read directory: %s' % dir)


def get_T_list_from_range(t_min: int, t_max: int, n_point: int = 8) -> [int]:
    t_span = t_max - t_min
    if t_max == t_min:
        return [t_min]
    if t_span <= 5:
        return [t_min, t_max]

    T_list = [t_min]
    while True:
        interval = t_span / (n_point - 1)
        if interval >= 5:
            break
        n_point -= 1

    for i in range(1, n_point):
        T_list.append(round(t_min + i * interval))

    return T_list


def get_T_list_VLE_from_range(t_min: int, t_max: int, n_point: int = 8) -> [int]:
    t_span = t_max - t_min
    if t_max == t_min:
        return [t_min]
    if t_span <= 5:
        return [t_min, t_max]

    T_list = [t_min]

    # TODO n_point=8
    interval = t_span / 13

    T_list.append(round(t_min + 4 * interval))
    T_list.append(round(t_min + 7 * interval))
    T_list.append(round(t_min + 9 * interval))
    T_list.append(round(t_min + 10 * interval))
    T_list.append(round(t_min + 11 * interval))
    T_list.append(round(t_min + 12 * interval))
    T_list.append(round(t_min + 13 * interval))

    return T_list


def get_P_list_from_range(p_min, p_max, multiple=(5,)) -> [int]:
    p_min = max(p_min, 1)
    p_max = max(p_max, 1)

    multiple = list(multiple)
    if 1 not in multiple:
        multiple.append(1)
    multiple.sort()

    magnitude_min = int(math.log10(p_min))
    magnitude_max = math.ceil(math.log10(p_max))

    P_list = []
    for i in range(magnitude_min, magnitude_max):
        for m in multiple:
            P = 10 ** i * m
            if P <= p_max:
                P_list.append(P)
    if p_max not in P_list:
        P_list.append(p_max)
    return P_list


def get_TP_corner(TP_list: [tuple]) -> [tuple]:
    TP_corner = []
    for t, p in TP_list:
        p_list = [TP[1] for TP in TP_list if TP[0] == t]
        if min(p_list) < p and max(p_list) > p:
            continue
        t_list = [TP[0] for TP in TP_list if TP[1] == p]
        if min(t_list) < t and max(t_list) > t:
            continue
        else:
            TP_corner.append((t, p))
    return TP_corner


def TP_in_range(TP, TP_corner: [tuple]) -> bool:
    ### TODO To implement
    return False


def create_mol_from_smiles(smiles: str, minimize=True, pdb_out: str = None, mol2_out: str = None, resname: str = None):
    # TODO resname only set for mol2_out
    import pybel
    from .saved_mol2 import smiles_mol2_dict
    try:
        py_mol = pybel.readstring('smi', smiles)
    except:
        raise OpenBabelError('Cannot create molecule from SMILES')

    canSMILES = py_mol.write('can').strip()
    saved_mol2 = smiles_mol2_dict.get(canSMILES)
    if saved_mol2 is not None:
        py_mol = next(pybel.readfile('mol2', saved_mol2))
    else:
        py_mol.addh()
        py_mol.make3D()
        if minimize:
            py_mol.localopt()

    if resname is not None:
        obmol = py_mol.OBMol
        res = obmol.GetResidue(0)
        # if res == None:
        #     res  = obmol.NewResidue()
        #     for i in range(obmol.NumAtoms()):
        #         obatom = obmol.GetAtomById(i)
        #         obatom.SetResidue(res)
        if res is not None:
            res.SetName('UNL')

    if pdb_out is not None:
        py_mol.write('pdb', pdb_out, overwrite=True)
    if mol2_out is not None:
        py_mol.write('mol2', mol2_out, overwrite=True)
        if resname is not None:
            with open(mol2_out) as f:
                content = f.read()
            content = content.replace('UNL', resname[:3])
            with open(mol2_out, 'w') as f:
                f.write(content)
    return py_mol


def generate_conformers(py_mol, number: int, redundant: int = 0):
    import openbabel as ob
    ff = ob.OBForceField.FindForceField('mmff94')

    smiles = py_mol.write('can').strip()
    py_mol.localopt()
    x_list = []
    for atom in py_mol.atoms:
        for x in atom.coords:
            x_list.append(x)
    xmin, xmax = min(x_list), max(x_list)
    xspan = xmax - xmin

    conformers = []
    for i in range(number + redundant):
        conformer = create_mol_from_smiles(smiles, minimize=False)

        for atom in conformer.atoms:
            obatom = atom.OBAtom
            random_coord = [(random.random() * xspan + xmin) * k for k in [2, 1, 0.5]]
            obatom.SetVector(*random_coord)

        conformer.localopt()
        ff.Setup(conformer.OBMol)
        conformer.OBMol.SetEnergy(ff.Energy())
        conformers.append(conformer)

    conformers.sort(key=lambda x: x.energy)
    return conformers[:number]


def is_alkane(py_mol) -> bool:
    import pybel
    from .formula import Formula
    atom_set = set(Formula(py_mol.formula).atomdict.keys())
    if atom_set != {'C', 'H'}:
        return False

    for s in ['[CX2]', '[CX3]', 'c', '[#6;v0,v1,v2,v3]']:
        if pybel.Smarts(s).findall(py_mol) != []:
            return False
    else:
        return True


def estimate_density_from_formula(f) -> float:
    # unit: g/mL
    from .formula import Formula
    formula = Formula.read(f)
    string = formula.to_str()
    density = {
        'H2': 0.07,
        'He': 0.15,
    }
    if string in density.keys():
        return density.get(string)

    nAtoms = formula.n_heavy + formula.n_h
    nC = formula.atomdict.get('C') or 0
    nH = formula.atomdict.get('H') or 0
    nO = formula.atomdict.get('O') or 0
    nN = formula.atomdict.get('N') or 0
    nS = formula.atomdict.get('S') or 0
    nF = formula.atomdict.get('F') or 0
    nCl = formula.atomdict.get('Cl') or 0
    nBr = formula.atomdict.get('Br') or 0
    nI = formula.atomdict.get('I') or 0
    nOther = nAtoms - nC - nH - nO - nN - nS - nF - nCl - nBr - nI
    return (1.175 * nC + 0.572 * nH + 1.774 * nO + 1.133 * nN + 2.184 * nS
            + 1.416 * nF + 2.199 * nCl + 5.558 * nBr + 7.460 * nI
            + 0.911 * nOther) / nAtoms


def n_diff_lines(f1: str, f2: str):
    with open(f1) as f:
        l1 = f.readlines()
    with open(f2) as f:
        l2 = f.readlines()

    n1 = len(l1)
    n2 = len(l2)

    n_diff = 0
    for i in range(min(n1, n2)):
        if l1[i].strip() != l2[i].strip():
            n_diff += 1
    n_diff += abs(n1 - n2)
    return n_diff


def get_last_line(filename):
    # TODO implement windows version
    cmd = 'tail -n 1 %s' % filename
    try:
        out = subprocess.check_output(cmd.split()).decode()
    except:
        raise Exception('Cannot open file: %s' % filename)

    try:
        string = out.splitlines()[-1]
    except:
        string = ''
    return string
