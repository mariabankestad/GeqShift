import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from typing import Callable, Optional
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce
from torch_geometric.nn import radius_graph
import json
from torch_geometric.data import InMemoryDataset



types = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'P': 6, 'S': 7, 'Cl': 8, 'Br':9}
bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}
hybid = {HybridizationType.UNSPECIFIED: 0,HybridizationType.S:1, HybridizationType.SP: 2, HybridizationType.SP2: 3 ,   HybridizationType.SP3: 4,
HybridizationType.SP3D:5 }


class Carbons13C(InMemoryDataset):


    def __init__(self, root: str, data_list, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def processed_file_names(self) -> str:
        return 'data_v3.pt'

    def process(self):

        data_list = []
        for mol in self.data_list:
            mol_name = mol.GetProp("_Name")
            nmr_spec = mol.GetProp("13C Spectrum")
            nmr_spec = eval(nmr_spec)
            type_idx_list = []
            nmr_list = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx_list.append(types[atom.GetSymbol()])
                num_hs.append(atom.GetTotalNumHs())
                if atom.GetIdx() in nmr_spec:
                    nmr_list.append(nmr_spec[atom.GetIdx()])
                else:
                    nmr_list.append(-1)


            type_idx = torch.tensor(type_idx_list, dtype=torch.float).reshape(-1,1)
            num_hs = torch.tensor(num_hs, dtype=torch.float).reshape(-1,1)
            nmr_list = torch.tensor(nmr_list, dtype=torch.float).reshape(-1,1)


            x = torch.cat([type_idx, num_hs, nmr_list], dim=-1)

            row, col = [], []
            bond_attr = []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start]
                col += [end]
                bond_attr +=  [bonds[bond.GetBondType()]]
            bond_attr = torch.tensor(bond_attr, dtype=torch.long).flatten()
            edge_index_b = torch.tensor([row, col], dtype=torch.long)
            edge_index_b, bond_attr = to_undirected(edge_index_b,bond_attr)
            


            
            positions = torch.zeros((x.size(0),3))

            positions = torch.tensor(positions, dtype = torch.float)
            edge_index_r = radius_graph(positions, r=1000.)
            edge_index_r = to_undirected(edge_index_r)
            edge_attr_r = torch.zeros(edge_index_r.size(1))
            edge_index = torch.column_stack([edge_index_b,edge_index_r])
            edge_attr = torch.cat([bond_attr, edge_attr_r])
            edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce='max')

            data = Data()
            data.x = x
            data.edge_attr = edge_attr
            data.edge_index = edge_index
            data.pos = positions
            data.name = mol_name
            data.smiles = Chem.MolToSmiles(mol)


            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
