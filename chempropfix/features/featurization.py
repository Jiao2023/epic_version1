# 如果bond feature包括atom feature，则需要修改两个地方 和 set_polymer

from typing import List, Tuple, Union
from itertools import zip_longest
from copy import deepcopy
from collections import Counter
import logging
# from .utils import get_distance_and_path
from rdkit import Chem
import torch
import numpy as np
from chempropfix.rdkit import make_mol
from torch_geometric.data import Data
import pdb

is_wDMPNN = False
def set_wDMPNN_mode(mode: bool) -> None:
    global is_wDMPNN
    is_wDMPNN = mode
    if is_wDMPNN:
        print(f"启动wDMPNN模式(节点信息包含边信息)")
    else:
        print(f"未启动wDMPNN模式(节点信息包含边信息)")


class Featurization_parameters:
    """
    A class holding molecule featurization parameters as attributes.
    """
    def __init__(self) -> None:

        # Atom feature sizes
        self.MAX_ATOMIC_NUM = 100
        self.ATOM_FEATURES = {
            'atomic_num': list(range(self.MAX_ATOMIC_NUM)),
            'degree': [0, 1, 2, 3, 4, 5],
            'formal_charge': [-1, -2, 1, 2, 0],
            'chiral_tag': [0, 1, 2, 3],
            'num_Hs': [0, 1, 2, 3, 4],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],
        }

        # Distance feature sizes
        self.PATH_DISTANCE_BINS = list(range(10))
        self.THREE_D_DISTANCE_MAX = 20
        self.THREE_D_DISTANCE_STEP = 1
        self.THREE_D_DISTANCE_BINS = list(range(0, self.THREE_D_DISTANCE_MAX + 1, self.THREE_D_DISTANCE_STEP))

        # len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
        self.ATOM_FDIM = sum(len(choices) + 1 for choices in self.ATOM_FEATURES.values()) + 2 
        self.EXTRA_ATOM_FDIM = 0
        self.BOND_FDIM = 14
        self.EXTRA_BOND_FDIM = 0
        self.REACTION_MODE = None
        self.EXPLICIT_H = False
        self.REACTION = False
        self.POLYMER = False
        self.ADDING_H = False
        self.GRAPH_CACHE_PATH = None

        self.KEEP_ATOM_MAP = False

# Create a global parameter object for reference throughout this module
PARAMS = Featurization_parameters()


def reset_featurization_parameters(logger: logging.Logger = None) -> None:
    """
    Function resets feature parameter values to defaults by replacing the parameters instance.
    """
    if logger is not None:
        debug = logger.debug
    else:
        debug = print
    debug('Setting molecule featurization parameters to default.')
    global PARAMS
    PARAMS = Featurization_parameters()


def get_atom_fdim(overwrite_default_atom: bool = False) -> int:
    """
    Gets the dimensionality of the atom feature vector.

    :param overwrite_default_atom: Whether to overwrite the default atom descriptors
    :return: The dimensionality of the atom feature vector.
    """
    return (not overwrite_default_atom) * PARAMS.ATOM_FDIM + PARAMS.EXTRA_ATOM_FDIM

def get_graph_cache_path()->str:
    return PARAMS.GRAPH_CACHE_PATH

def set_explicit_h(explicit_h: bool) -> None:
    """
    Sets whether RDKit molecules will be constructed with explicit Hs.

    :param explicit_h: Boolean whether to keep explicit Hs from input.
    """
    PARAMS.EXPLICIT_H = explicit_h

def set_adding_hs(adding_hs: bool) -> None:
    """
    Sets whether RDKit molecules will be constructed with adding the Hs to them.

    :param adding_hs: Boolean whether to add Hs to the molecule.
    """
    PARAMS.ADDING_H = adding_hs

def set_keeping_atom_map(keeping_atom_map: bool) -> None:
    """
    Sets whether RDKit molecules keep the original atom mapping.

    :param keeping_atom_map: Boolean whether to keep the original atom mapping.
    """
    PARAMS.KEEP_ATOM_MAP = keeping_atom_map

def set_polymer(polymer: bool) -> None:
    """
    Sets whether RDKit molecules are two monomers of a co-polymer.

    :param polymer: Boolean whether input is two monomer units of a co-polymer.
    """
    PARAMS.POLYMER = polymer
# set_polymer(True)

def set_reaction(reaction: bool, mode: str) -> None:
    """
    Sets whether to use a reaction or molecule as input and adapts feature dimensions.
 
    :param reaction: Boolean whether to except reactions as input.
    :param mode: Reaction mode to construct atom and bond feature vectors.

    """
    PARAMS.REACTION = reaction
    if reaction:
        PARAMS.EXTRA_ATOM_FDIM = PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM -1
        PARAMS.EXTRA_BOND_FDIM = PARAMS.BOND_FDIM
        PARAMS.REACTION_MODE = mode        

def set_graph_cache_path(path:str)->None:
    PARAMS.GRAPH_CACHE_PATH = path
        
def is_explicit_h(is_mol: bool = True) -> bool:
    r"""Returns whether to retain explicit Hs (for reactions only)"""
    if not is_mol:
        return PARAMS.EXPLICIT_H
    return False


def is_adding_hs(is_mol: bool = True) -> bool:
    r"""Returns whether to add explicit Hs to the mol (not for reactions)"""
    if is_mol:
        return PARAMS.ADDING_H
    return False

def is_reaction(is_mol: bool = True) -> bool:
    r"""Returns whether to use reactions as input"""
    if is_mol:
        return False
    if PARAMS.REACTION: #(and not is_mol, checked above)
        return True
    return False


def is_keeping_atom_map(is_mol: bool = True) -> bool:
    r"""Returns whether to keep the original atom mapping (not for reactions)"""
    if is_mol:
        return PARAMS.KEEP_ATOM_MAP
    return True

def is_polymer() -> bool:
    r"""Returns whether to the molecule is a polymer"""
    return PARAMS.POLYMER


def reaction_mode() -> str:
    r"""Returns the reaction mode"""
    return PARAMS.REACTION_MODE


def set_extra_atom_fdim(extra):
    """Change the dimensionality of the atom feature vector."""
    PARAMS.EXTRA_ATOM_FDIM = extra


def get_bond_fdim(atom_messages: bool = False,
                  overwrite_default_bond: bool = False,
                  overwrite_default_atom: bool = False) -> int:
    """
    Gets the dimensionality of the bond feature vector.

    :param atom_messages: Whether atom messages are being used. If atom messages are used,
                          then the bond feature vector only contains bond features.
                          Otherwise it contains both atom and bond features.
    :param overwrite_default_bond: Whether to overwrite the default bond descriptors
    :param overwrite_default_atom: Whether to overwrite the default atom descriptors
    :return: The dimensionality of the bond feature vector.
    """
    # 如果bond feature包括atom feature则不需要注释
    if is_wDMPNN:
            return (not overwrite_default_bond) * PARAMS.BOND_FDIM + PARAMS.EXTRA_BOND_FDIM \
            + \
           (not atom_messages) * get_atom_fdim(overwrite_default_atom=overwrite_default_atom)
    else:
        
        return (not overwrite_default_bond) * PARAMS.BOND_FDIM + PARAMS.EXTRA_BOND_FDIM \
            #+ \
            # (not atom_messages) * get_atom_fdim(overwrite_default_atom=overwrite_default_atom)


def set_extra_bond_fdim(extra):
    """Change the dimensionality of the bond feature vector."""
    PARAMS.EXTRA_BOND_FDIM = extra


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) + \
            onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES['degree']) + \
            onek_encoding_unk(atom.GetFormalCharge(), PARAMS.ATOM_FEATURES['formal_charge']) + \
            onek_encoding_unk(int(atom.GetChiralTag()), PARAMS.ATOM_FEATURES['chiral_tag']) + \
            onek_encoding_unk(int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES['num_Hs']) + \
            onek_encoding_unk(int(atom.GetHybridization()), PARAMS.ATOM_FEATURES['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0] + \
            [atom.GetMass() * 0.01]  # scaled to about the same range as other features
        if functional_groups is not None:
            features += functional_groups
    return features


def atom_features_zeros(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom containing only the atom number information.

    :param atom: An RDKit atom.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, PARAMS.ATOM_FEATURES['atomic_num']) + \
            [0] * (PARAMS.ATOM_FDIM - PARAMS.MAX_ATOMIC_NUM - 1) #set other features to zero
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (PARAMS.BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def map_reac_to_prod(mol_reac: Chem.Mol, mol_prod: Chem.Mol):
    """
    Build a dictionary of mapping atom indices in the reactants to the products.

    :param mol_reac: An RDKit molecule of the reactants.
    :param mol_prod: An RDKit molecule of the products.
    :return: A dictionary of corresponding reactant and product atom indices.
    """
    only_prod_ids = []
    prod_map_to_id = {}
    mapnos_reac = set([atom.GetAtomMapNum() for atom in mol_reac.GetAtoms()]) 
    for atom in mol_prod.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            prod_map_to_id[mapno] = atom.GetIdx()
            if mapno not in mapnos_reac:
                only_prod_ids.append(atom.GetIdx()) 
        else:
            only_prod_ids.append(atom.GetIdx())
    only_reac_ids = []
    reac_id_to_prod_id = {}
    for atom in mol_reac.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            try:
                reac_id_to_prod_id[atom.GetIdx()] = prod_map_to_id[mapno]
            except KeyError:
                only_reac_ids.append(atom.GetIdx())
        else:
            only_reac_ids.append(atom.GetIdx())
    return reac_id_to_prod_id, only_prod_ids, only_reac_ids


def tag_atoms_in_repeating_unit(mol):
    """
    Tags atoms that are part of the core units, as well as atoms serving to identify attachment points. In addition,
    create a map of bond types based on what bonds are connected to R groups in the input.
    """
    atoms = [a for a in mol.GetAtoms()]
    neighbor_map = {}  # map R group to index of atom it is attached to
    r_bond_types = {}  # map R group to bond type

    # go through each atoms and: (i) get index of attachment atoms, (ii) tag all non-R atoms
    for atom in atoms:
        # if R atom
        atom.SetBoolProp('termini',False)
        if '*' in atom.GetSmarts():
            # get index of atom it is attached to
            neighbors = atom.GetNeighbors()
            assert len(neighbors) == 1
            neighbor_idx = neighbors[0].GetIdx()
            mol.GetAtomWithIdx(neighbor_idx).SetBoolProp('termini', True)
            r_tag = atom.GetSmarts().strip('[]').replace(':', '')  # *1, *2, ...
            neighbor_map[r_tag] = neighbor_idx
            # tag it as non-core atom
            atom.SetBoolProp('core', False)
            # create a map R --> bond type
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor_idx)
            r_bond_types[r_tag] = bond.GetBondType()
        # if not R atom
        else:
            # tag it as core atom
            atom.SetBoolProp('core', True)

    # use the map created to tag attachment atoms
    for atom in atoms:
        if atom.GetIdx() in neighbor_map.values():
            r_tags = [k for k, v in neighbor_map.items() if v == atom.GetIdx()]
            atom.SetProp('R', ''.join(r_tags))
        else:
            atom.SetProp('R', '')

    return mol, r_bond_types


def remove_wildcard_atoms(rwmol):
    indices = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
    while len(indices) > 0:
        rwmol.RemoveAtom(indices[0])
        indices = [a.GetIdx() for a in rwmol.GetAtoms() if '*' in a.GetSmarts()]
    Chem.SanitizeMol(rwmol, Chem.SanitizeFlags.SANITIZE_ALL)
    return rwmol


def parse_polymer_rules(rules):
    polymer_info = []
    counter = Counter()  # used for validating the input

    # check if deg of polymerization is provided
    if '~' in rules[-1]:
        Xn = float(rules[-1].split('~')[1])
        rules[-1] = rules[-1].split('~')[0]
    else:
        Xn = 1.

    for rule in rules:
        # handle edge case where we have no rules, and rule is empty string
        if rule == "":
            continue
        # QC of input string
        if len(rule.split(':')) != 3:
            raise ValueError(f'incorrect format for input information "{rule}"')
        idx1, idx2 = rule.split(':')[0].split('-')
        w12 = float(rule.split(':')[1])  # weight for bond R_idx1 -> R_idx2
        w21 = float(rule.split(':')[2])  # weight for bond R_idx2 -> R_idx1
        polymer_info.append((idx1, idx2, w12, w21))
        counter[idx1] += float(w21)
        counter[idx2] += float(w12)

    # validate input: sum of incoming weights should be one for each vertex
    for k, v in counter.items():
        if np.isclose(v, 1.0) is False:
            raise ValueError(f'sum of weights of incoming stochastic edges should be 1 -- found {v} for [*:{k}]')
    return polymer_info, 1. + np.log10(Xn)


class MolGraph :
    """
    A :class:`MolGraph` represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:

    * :code:`n_atoms`: The number of atoms in the molecule.
    * :code:`n_bonds`: The number of bonds in the molecule.
    * :code:`f_atoms`: A mapping from an atom index to a list of atom features.
    * :code:`f_bonds`: A mapping from a bond index to a list of bond features.
    * :code:`a2b`: A mapping from an atom index to a list of incoming bond indices.
    * :code:`b2a`: A mapping from a bond index to the index of the atom the bond originates from.
    * :code:`b2revb`: A mapping from a bond index to the index of the reverse bond.
    * :code:`overwrite_default_atom_features`: A boolean to overwrite default atom descriptors.
    * :code:`overwrite_default_bond_features`: A boolean to overwrite default bond descriptors.
    """

    def __init__(self, mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
                 atom_features_extra: np.ndarray = None,
                 bond_features_extra: np.ndarray = None,
                 overwrite_default_atom_features: bool = False,
                 overwrite_default_bond_features: bool = False):
        """
        :param mol: A SMILES or an RDKit molecule.
        :param atom_features_extra: A list of 2D numpy array containing additional atom features to featurize the molecule
        :param bond_features_extra: A list of 2D numpy array containing additional bond features to featurize the molecule
        :param overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features instead of concatenating
        :param overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features instead of concatenating
        """
        self.is_mol = is_mol(mol)  
        self.is_reaction = is_reaction(self.is_mol)
        self.is_explicit_h = is_explicit_h(self.is_mol)
        self.is_adding_hs = is_adding_hs(self.is_mol)
        self.is_keeping_atom_map = is_keeping_atom_map(self.is_mol)
        self.reaction_mode = reaction_mode()
        
        if type(mol) == str:
            if self.is_reaction:
                mol = (make_mol(mol.split(">")[0], self.is_explicit_h, self.is_adding_hs),
                       make_mol(mol.split(">")[-1], self.is_explicit_h, self.is_adding_hs))
                # TODO: use BigSMILES notation as input for polymers with a dedicated parser
                mol = (make_polymer_mol(mol.split("|")[0], self.is_explicit_h, self.is_adding_hs,  # smiles
                                        fragment_weights=mol.split("|")[1:-1]),  # fraction of each fragment
                       mol.split("<")[1:])  # edges between fragments
        
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features

        if not self.is_reaction:
            self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]
            if atom_features_extra is not None:
                if overwrite_default_atom_features:
                    self.f_atoms = [descs.tolist() for descs in atom_features_extra]
                else:
                    self.f_atoms = [f_atoms + descs.tolist() for f_atoms, descs in zip(self.f_atoms, atom_features_extra)]
            self.n_atoms = len(self.f_atoms)
            if atom_features_extra is not None and len(atom_features_extra) != self.n_atoms:
                raise ValueError(f'The number of atoms in {Chem.MolToSmiles(mol)} is different from the length of '
                                 f'the extra atom features')
            # Initialize atom to bond mapping for each atom
            for _ in range(self.n_atoms):
                self.a2b.append([])            
            # Initialize f_bonds to real bonds mapping for each bond
            self.b2br = np.zeros([len(mol.GetBonds()), 2])
            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)
                    if bond is None:
                        continue
                    f_bond = bond_features(bond)
                    if bond_features_extra is not None:
                        descr = bond_features_extra[bond.GetIdx()].tolist()
                        if overwrite_default_bond_features:
                            f_bond = descr
                        else:
                            f_bond += descr

                    self.f_bonds.append(f_bond)
                    self.f_bonds.append(f_bond)
                    
                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    # 上面这些操作是为了存储双向边
                    self.b2br[bond.GetIdx(), :] = [self.n_bonds, self.n_bonds + 1]
                    self.n_bonds += 2

            if bond_features_extra is not None and len(bond_features_extra) != self.n_bonds / 2:
                raise ValueError(f'The number of bonds in {Chem.MolToSmiles(mol)} is different from the length of '
                                 f'the extra bond features')
        

class BatchMolGraph:
    """
    A :class:`BatchMolGraph` represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a :class:`MolGraph` plus:

    * :code:`atom_fdim`: The dimensionality of the atom feature vector.
    * :code:`bond_fdim`: The dimensionality of the bond feature vector (technically the combined atom/bond features).
    * :code:`a_scope`: A list of tuples indicating the start and end atom indices for each molecule.
    * :code:`b_scope`: A list of tuples indicating the start and end bond indices for each molecule.
    * :code:`max_num_bonds`: The maximum number of bonds neighboring an atom in this batch.
    * :code:`b2b`: (Optional) A mapping from a bond index to incoming bond indices.
    * :code:`a2a`: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph]):
        r"""
        :param mol_graphs: A list of :class:`MolGraph`\ s from which to construct the :class:`BatchMolGraph`.
        """
        self.mol_graphs = mol_graphs
        self.overwrite_default_atom_features = mol_graphs[0].overwrite_default_atom_features
        self.overwrite_default_bond_features = mol_graphs[0].overwrite_default_bond_features
        self.atom_fdim = get_atom_fdim(overwrite_default_atom=self.overwrite_default_atom_features)
        self.bond_fdim = get_bond_fdim(overwrite_default_bond=self.overwrite_default_bond_features,
                                       overwrite_default_atom=self.overwrite_default_atom_features)

        
        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule
        
        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond feature
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)
            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        
        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages
        self.b2br = None  # only needed in predictions of atomic/bond targets

    def get_components(self, atom_messages: bool = False):
        """
        Returns the components of the :class:`BatchMolGraph`.

        The returned components are, in order:

        * :code:`f_atoms`
        * :code:`f_bonds`
        * :code:`a2b`
        * :code:`b2a`
        * :code:`b2revb`
        * :code:`a_scope`
        * :code:`b_scope`

        :param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond feature
                              vector to contain only bond features rather than both atom and bond features.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, graph structure,
                 and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).
        """
        if atom_messages:
            f_bonds = self.f_bonds[:, -get_bond_fdim(atom_messages=atom_messages,
                                                     overwrite_default_atom=self.overwrite_default_atom_features,
                                                     overwrite_default_bond=self.overwrite_default_bond_features):]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds,self.a2b, self.b2a, self.b2revb,self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each atom index to all the neighboring atom indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a

    def get_b2br(self) -> torch.Tensor:
        """
        Computes (if necessary) and returns a mapping from f_bonds to real bonds in molecule recorded in targets.

        :return: A PyTorch tensor containing the mapping from f_bonds to real bonds in molecule recorded in targets.
        """
        if self.b2br is None:
            n_bonds = 1 # number of bonds (start at 1 b/c need index 0 as padding)
            b2br = []
            for mol_graph in self.mol_graphs:
                b2br.append(mol_graph.b2br + n_bonds)
                n_bonds += mol_graph.n_bonds
            b2br = np.concatenate(b2br, axis=0)
            self.b2br = torch.tensor(b2br, dtype=torch.long)

        return self.b2br

def mol2graph(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
              mtypes:List[int],
              atom_features_batch: List[np.array] = (None,),
              bond_features_batch: List[np.array] = (None,),
               overwrite_default_atom_features: bool = False,
             overwrite_default_bond_features: bool = False
              ) -> BatchMolGraph:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.

    :param mols: A list of SMILES or a list of RDKit molecules.
    :param atom_features_batch: A list of 2D numpy array containing additional atom features to featurize the molecule
    :param bond_features_batch: A list of 2D numpy array containing additional bond features to featurize the molecule
    :param overwrite_default_atom_features: Boolean to overwrite default atom descriptors by atom_descriptors instead of concatenating
    :param overwrite_default_bond_features: Boolean to overwrite default bond descriptors by bond_descriptors instead of concatenating
    :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.
    """
    return BatchMolGraph([MolGraph(mol,mtype, af, bf, 
                                   overwrite_default_atom_features=overwrite_default_atom_features,
                                   overwrite_default_bond_features=overwrite_default_bond_features)
                          for mol, af, bf, mtype
                          in zip_longest(mols, atom_features_batch, bond_features_batch,mtypes)])


def is_mol(mol: Union[str, Chem.Mol,Tuple[Chem.Mol, Chem.Mol]]) -> bool:
    """Checks whether an input is a molecule or a reaction

    :param mol: str, RDKIT molecule or tuple of molecules.
    :return: Whether the supplied input corresponds to a single molecule.
    """

    if isinstance(mol, str) and ">" not in mol:
        return True
    elif isinstance(mol, Chem.Mol):
        return True
    return False