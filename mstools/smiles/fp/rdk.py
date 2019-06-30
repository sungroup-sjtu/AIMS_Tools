import sys
import pathlib
import numpy as np
import networkx as nx
from pathlib import Path

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect

from . import Fingerprint
from .drawmorgan import DrawMorganBit


class ECFP4Indexer(Fingerprint):
    name = 'ecfp4'

    def __init__(self):
        super().__init__()
        self.n_bits = 1024

    def index(self, smiles):
        rdk_mol = Chem.MolFromSmiles(smiles)
        return np.array(
            list(map(int, Chem.GetMorganFingerprintAsBitVect(rdk_mol, radius=2, nBits=self.n_bits))))

    def index_list(self, smiles_list):
        return [self.index(s) for s in smiles_list]


class MorganCountIndexer(Fingerprint):
    name = 'morgan'

    def __init__(self, fp_time_limit=200):
        super().__init__()
        self.radius = 2
        self.fp_time_limit = fp_time_limit
        self.svg_dir: Path = None

    def index(self, smiles):
        raise Exception('Use index_list() for this indexer')

    def index_list(self, smiles_list):
        rdkfp_list = []
        identifiers = []
        fpsvg_dict = {}
        fpsmi_dict = {}
        bits_list = []

        print('Calculate with RDKit...')
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                sys.stdout.write('\r\t%i' % i)

            rdk_mol = Chem.MolFromSmiles(smiles)
            info = dict()
            rdkfp: UIntSparseIntVect = Chem.GetMorganFingerprint(rdk_mol, radius=self.radius, bitInfo=info)
            rdkfp_list.append(rdkfp)
            identifiers += list(rdkfp.GetNonzeroElements().keys())

            if self.svg_dir is not None:
                for idx in info.keys():
                    if idx not in fpsvg_dict:
                        fpsvg_dict[idx] = DrawMorganBit(rdk_mol, idx, info)

                        root, radius = info[idx][0]
                        if radius == 0:
                            id_atoms = [root]
                        else:
                            id_bonds = Chem.FindAtomEnvironmentOfRadiusN(rdk_mol, radius, root)  # args: mol, radius, atomId
                            id_atoms = set()
                            for bid in id_bonds:
                                id_atoms.add(rdk_mol.GetBondWithIdx(bid).GetBeginAtomIdx())
                                id_atoms.add(rdk_mol.GetBondWithIdx(bid).GetEndAtomIdx())
                            id_atoms = list(id_atoms)
                        smi = Chem.MolFragmentToSmiles(rdk_mol, atomsToUse=id_atoms, rootedAtAtom=root, isomericSmiles=False)
                        fpsmi_dict[idx] = smi

        print('\nFilter identifiers...')
        identifiers = set(identifiers)
        print('%i identifiers total' % len(identifiers))
        idx_times = dict([(id, 0) for id in identifiers])
        for rdkfp in rdkfp_list:
            for idx in rdkfp.GetNonzeroElements().keys():
                idx_times[idx] += 1
        self.bit_count = dict([(idx, 0) for idx, times in sorted(idx_times.items(), key=lambda x: x[1], reverse=True) if times >= self.fp_time_limit])
        print('%i identifiers appears in more than %i molecules saved' % (len(self.idx_list), self.fp_time_limit))

        for rdkfp in rdkfp_list:
            bits = [rdkfp.GetNonzeroElements().get(idx, 0) for idx in self.idx_list]
            bits_list.append(np.array(bits))

        if self.svg_dir is not None:
            if not self.svg_dir.exists():
                self.svg_dir.mkdir()
            print('Save figures...')
            for idx in self.idx_list:
                figure = '%i-%i-%s.svg' % (idx_times[idx], idx, fpsmi_dict[idx])
                with open(pathlib.Path(self.svg_dir, figure), 'w') as f:
                    f.write(fpsvg_dict[idx])

        return bits_list


class Morgan1CountIndexer(MorganCountIndexer):
    name = 'morgan1'

    def __init__(self, fp_time_limit=200):
        super().__init__()
        self.radius = 1
        self.fp_time_limit = fp_time_limit


class PredefinedMorganCountIndexer(Fingerprint):
    name = 'predefinedmorgan'

    def __init__(self):
        super().__init__()
        self.radius = 2
        self.use_pre_idx_list = 'morgan'
        self.pre_idx_list = []

    def index(self, smiles):
        rdk_mol = Chem.MolFromSmiles(smiles)
        fp: UIntSparseIntVect = Chem.GetMorganFingerprint(rdk_mol, radius=self.radius)
        return np.array([fp.GetNonzeroElements().get(int(idx), 0) for idx in self.pre_idx_list])

    def index_list(self, smiles_list):
        return [self.index(s) for s in smiles_list]


class PredefinedMorgan1CountIndexer(PredefinedMorganCountIndexer):
    name = 'predefinedmorgan1'

    def __init__(self):
        super().__init__()
        self.radius = 1
        self.use_pre_idx_list = 'morgan1'

atom_dict = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5:'B',
    6: 'C', 7: 'N', 8: 'O', 9: 'F', 10:'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15:'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20:'Ga',
    35: 'Br', 53: 'I',
}

class Vertice:
    def __init__(self, atom_num, formal_charge, idx):
        self.atom_num = atom_num
        self.formal_charge = formal_charge
        self.atom_name = atom_dict.get(atom_num)
        if self.atom_name == None:
            print('atom %i is not in the atom_dict' % atom_num)

        # infomation in graph
        self.idx = idx

    def __str__(self):
        return str(self.atom_name)

    def __eq__(self, other):
        return [self.atom_num, self.formal_charge] == [other.atom_num, other.formal_charge]

    def __hash__(self):
        return hash(id(self))

class GraphPath:
    def __init__(self, vertices, weight):
        self.vertices = vertices
        self.weight = weight

    def __eq__(self, other):
        return [self.vertices, self.weight] == [other.vertices, other.weight]
        '''
        if len(self.vertices) != len(other.vertices):
            return False
        for i in range(len(self.vertices)):
            if self.vertices[i].atom_num != other.vertices[i].atom_num or self.vertices[i].formal_charge != other.vertices[i].formal_charge:
                return False
        return self.weight == other.weight
'''
    def __hash__(self):
        return 19930319
'''
def get_unique_list(input_list):
    result_list = []
    for info in input_list:
        if info not in result_list:
            result_list.append(info)
    return result_list
'''
class PathIndexer(Fingerprint):
    name = 'path'

    def __init__(self, fp_time_limit = 200):
        super().__init__()
        self.fp_time_limit = fp_time_limit

    def index_list(self, smiles_list):
        print('Calculate Path Indexer with RDKit...')
        graph_list = []
        graph_path_list = []
        bits_list = []
        unique_path_list = set()
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                sys.stdout.write('\r\tmolecule number %i' % i)

            G = self.get_graph_from_smiles(smiles)
            graph_list.append(G)

            # get path infomation
            p_list = []
            for j in range(len(list(G.node))):
                for k in range(j, len(list(G.node))):
                    paths = nx.all_simple_paths(G, source=list(G.node)[j], target=list(G.node)[k])
                    for path in paths:
                        weight = []
                        for l in range(len(path)-1):
                            weight.append(G.get_edge_data(path[l], path[l+1]).get('weight'))
                        graph_path = GraphPath(path, weight)
                        p_list.append(graph_path)
            unique_path_list |= set(p_list)
            graph_path_list.append(p_list)

        print('\nFilter paths...')
        print('%i distinct paths total' % len(unique_path_list))

        idx_times = dict([(id, 0) for id in unique_path_list])
        for i, p_list in enumerate(graph_path_list):
            if i % 100 == 0:
                sys.stdout.write('\r\tmolecule number %i' % i)
            for idx in set(p_list):
                idx_times[idx] += 1
        self.bit_count = dict([(idx, 0) for idx, times in sorted(idx_times.items(), key=lambda x: x[1], reverse=True) if
                               times >= self.fp_time_limit])
        print('%i identifiers appears in more than %i molecules saved' % (len(self.idx_list), self.fp_time_limit))

        for p_list in graph_path_list:
            bits = [p_list.count(idx) for idx in self.idx_list]
            bits_list.append(np.array(bits))

        return bits_list

    def get_graph_from_smiles(self, smiles):
        rdk_mol = Chem.MolFromSmiles(smiles)
        rdk_mol = Chem.AddHs(rdk_mol)
        atom_number = rdk_mol.GetNumAtoms()
        G = nx.Graph()
        for atom in rdk_mol.GetAtoms():
            v = Vertice(atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIdx())
            G.add_node(v)
        for i in range(atom_number):
            for j in range(i + 1, atom_number):
                bond_ij = rdk_mol.GetBondBetweenAtoms(i, j)
                if bond_ij != None:
                    G.add_edge(list(G.nodes)[i], list(G.nodes)[j], weight=bond_ij.GetBondTypeAsDouble())
        return G


class PathNOHIndexer(Fingerprint):
    name = 'pathnoh'

    def __init__(self, fp_time_limit = 200):
        super().__init__()
        self.fp_time_limit = fp_time_limit

    def index_list(self, smiles_list):
        print('Calculate Path Indexer with RDKit...')
        graph_list = []
        graph_path_list = []
        bits_list = []
        unique_path_list = set()
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                sys.stdout.write('\r\tmolecule number %i' % i)

            G = self.get_graph_from_smiles(smiles)
            graph_list.append(G)

            # get path infomation
            p_list = []
            for j in range(len(list(G.node))):
                for k in range(j, len(list(G.node))):
                    paths = nx.all_simple_paths(G, source=list(G.node)[j], target=list(G.node)[k])
                    for path in paths:
                        weight = []
                        for l in range(len(path)-1):
                            weight.append(G.get_edge_data(path[l], path[l+1]).get('weight'))
                        graph_path = GraphPath(path, weight)
                        p_list.append(graph_path)
            unique_path_list |= set(p_list)
            graph_path_list.append(p_list)

        print('\nFilter paths...')
        print('%i distinct paths total' % len(unique_path_list))

        idx_times = dict([(id, 0) for id in unique_path_list])
        for i, p_list in enumerate(graph_path_list):
            if i % 100 == 0:
                sys.stdout.write('\r\tmolecule number %i' % i)
            for idx in set(p_list):
                idx_times[idx] += 1
        self.bit_count = dict([(idx, 0) for idx, times in sorted(idx_times.items(), key=lambda x: x[1], reverse=True) if
                               times >= self.fp_time_limit])
        print('%i identifiers appears in more than %i molecules saved' % (len(self.idx_list), self.fp_time_limit))

        for p_list in graph_path_list:
            bits = [p_list.count(idx) for idx in self.idx_list]
            bits_list.append(np.array(bits))

        return bits_list

    def get_graph_from_smiles(self, smiles):
        rdk_mol = Chem.MolFromSmiles(smiles)
        atom_number = rdk_mol.GetNumAtoms()
        G = nx.Graph()
        for atom in rdk_mol.GetAtoms():
            v = Vertice(atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIdx())
            G.add_node(v)
        for i in range(atom_number):
            for j in range(i + 1, atom_number):
                bond_ij = rdk_mol.GetBondBetweenAtoms(i, j)
                if bond_ij != None:
                    G.add_edge(list(G.nodes)[i], list(G.nodes)[j], weight=bond_ij.GetBondTypeAsDouble())
        return G