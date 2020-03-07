import math

from ..utils import create_mol_from_smiles, estimate_density_from_formula
from ..wrapper import Packmol, DFF


class Simulation():
    def __init__(self, packmol=None, dff=None, packmol_bin=None, dff_root=None, dff_db=None, dff_table=None, jobmanager=None):
        if packmol is not None:
            self.packmol = packmol
        elif packmol_bin is not None:
            self.packmol = Packmol(packmol_bin=packmol_bin)
        if dff is not None:
            self.dff = dff
        elif dff_root is not None:
            self.dff = DFF(dff_root=dff_root, default_db=dff_db, default_table=dff_table)
        self.jobmanager = jobmanager
        self.procedure = None
        self.n_atom_default: int = 3000
        self.n_mol_default: int = 120

        self.n_mol_list: [int]
        self.msd = 'init.msd'
        self.pdb = 'init.pdb'
        self._single_msd = '_single.msd'
        self._single_pdb = '_single.pdb'

    def build(self):
        pass

    def prepare(self):
        pass

    def run(self):
        self.jobmanager.submit()

    def check_finished(self):
        pass

    def analyze(self):
        pass

    def clean(self):
        pass

    def post_process(self, **kwargs):
        pass

    def get_post_data(self, **kwargs):
        pass

    def set_system(self, smiles_list: [str], n_mol_list: [int] = None, n_atoms: int = None, n_mols: int = None, n_mol_ratio: [int] = None,
                   length: float = None, density: float = None, name_list: [str] = None):
        if type(smiles_list) != list:
            raise Exception('smiles_list should be list')
        self.smiles_list = smiles_list[:]
        self.pdb_list = []
        self.mol2_list = []
        n_components = len(smiles_list)
        n_atom_list = []  # number of atoms of each molecule
        molwt_list = []  # molecule weight of each molecule
        density_list = []  # estimated density of each molecule
        for i, smiles in enumerate(smiles_list):
            pdb = 'mol-%i.pdb' % i
            mol2 = 'mol-%i.mol2' % i
            if name_list is not None:
                resname = name_list[i]
            else:
                resname = 'MO%i' % i
            py_mol = create_mol_from_smiles(smiles, pdb_out=pdb, mol2_out=mol2, resname=resname)
            self.pdb_list.append(pdb)
            self.mol2_list.append(mol2)
            n_atom_list.append(len(py_mol.atoms))
            molwt_list.append(py_mol.molwt)
            density_list.append(estimate_density_from_formula(py_mol.formula) * 0.9)  # * 0.9, build box will be faster

        if n_mol_list is not None:
            self.n_mol_list = n_mol_list
        else:
            if n_mol_ratio is None:
                n_mol_ratio = [1] * n_components
            n_atom_all = sum([n_atom_list[i] * n for i, n in enumerate(n_mol_ratio)])
            if n_atoms is not None:
                self.n_atom_default = n_atoms
            if n_mols is not None:
                self.n_mol_default = n_mols
            n_atoms_from_mol = n_atom_all * math.ceil(self.n_mol_default / sum(n_mol_ratio))
            n_atoms = max(n_atoms_from_mol, self.n_atom_default)
            self.n_mol_list = [math.ceil(n_atoms / n_atom_all) * n for n in n_mol_ratio]

        mass = sum([molwt_list[i] * self.n_mol_list[i] for i in range(n_components)])

        if length is not None:
            self.length = length
            self.box = [length, length, length]
            self.vol = self.length ** 3
        else:
            if density is None:
                density = sum([density_list[i] * self.n_mol_list[i] for i in range(n_components)]) / sum(
                    self.n_mol_list)
            self.vol = 10 / 6.022 * mass / density
            self.length = self.vol ** (1 / 3)  # assume cubic box
            self.box = [self.length, self.length, self.length]
