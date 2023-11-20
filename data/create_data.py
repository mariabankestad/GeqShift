
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers
import pickle
import numpy as np
import torch
import os 
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_undirected
from CarbonDataset import Carbons13C
from HydrogenDataset import Hydrogens1H

from torch_geometric.data.collate import collate
from torch_geometric.loader import DataLoader

def get_cut_off_graph(edge_index, edge_attr, p, cut_off = 6.0):
    row, col = edge_index
    dist = torch.sqrt(torch.sum((p[row]- p[col])**2, dim = 1))
    mask = dist <= cut_off
    edge_index = edge_index[:,mask]
    edge_attr = edge_attr[mask]
    edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce  = "max")
    return edge_index, edge_attr

def generate_conformations(m_, nbr_confs = 10):
    params = Chem.rdDistGeom.ETKDGv3()
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.pruneRmsThresh = 0.001
    params.numThreads = 10
    params.enforceChirality = True
    params.maxAttempts = 10000
    Chem.SanitizeMol(m_)
    m_ = Chem.AddHs(m_, addCoords=True)

    em = Chem.rdDistGeom.EmbedMultipleConfs(m_, numConfs=nbr_confs*2, params=params)

    ps = AllChem.MMFFGetMoleculeProperties(m_, mmffVariant='MMFF94')
    
    energies = []
    for conf in m_.GetConformers():
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(m_, ps, confId=conf.GetId())
        
        if isinstance(ff, type(None)):
            continue 
        energy = ff.CalcEnergy()
        energies.append(energy)

    m_ = Chem.RemoveHs(m_)
    if em == -1:
            conformations = []            
            for i, c in enumerate(m_.GetConformers()):
                xyz = c.GetPositions()
                conformations.append(xyz)
                if i >9:
                    return conformations
    energies = np.array(energies)
    ind = energies.argsort()[:nbr_confs]
    energies = energies[ind]
    conformations = []
    for i, c in enumerate(m_.GetConformers()):

        if i not in ind:
            continue
        xyz = c.GetPositions()
        conformations.append(xyz)

    return conformations, energies, m_

def create_conformer(mols, nbr_confs = 10):
    for mol in mols:
        name = mol.GetProp("_Name")

        name = name.replace("->", "")
        name = name.replace("(","")
        name = name.replace(")","")

        path = "conformations//" + name + ".pickle"
        if os.path.isfile(path):
            continue
        conformations, _,_ = generate_conformations(mol,nbr_confs)
        with open(path, 'wb') as handle:
            pickle.dump(conformations, handle)

def train_test_split(mols, fold):
    N_t = len(mols)
    indices = range(N_t)
    indices_kfold = np.array_split(indices, 10)[fold]
    train_mask = np.ones(N_t, dtype=bool)
    train_mask[indices_kfold] = False
    test_mask = [not(a) for a in train_mask]
    train_datas = [data for data, m in zip(mols, train_mask) if m]
    test_datas = [data for data, m in zip(mols, test_mask) if m]   
    return  train_datas, test_datas

def add_conformations_to_train_dataset(dataset):
    new_dataset = []
    for d_ in dataset:
        name = d_.name
        name = name.replace("->","")
        name = name.replace(")","")
        name = name.replace("(","")
        conf_path = "conformations//" + name + ".pickle"
        #if not os.path.exists(file_path):
        #    continue

        with open(conf_path, 'rb') as handle:
            distencies = pickle.load(handle)
            for dis_ in distencies:
                new_data = d_.clone()
                new_data.pos = torch.tensor(dis_)
                edge_index, edge_attr = get_cut_off_graph(new_data.edge_index, new_data.edge_attr, new_data.pos ,cut_off=6.)
                new_data.edge_index = edge_index
                new_data.edge_attr = edge_attr
                new_dataset.append(new_data)
    new_dataset_ = InMemoryDataset()
    data, slices, _ = collate(
                new_dataset[0].__class__,
                data_list=new_dataset,
                increment=False,
                add_batch=False,
            )
    new_dataset_.data = data

    new_dataset_.slices = slices
    return new_dataset_

def add_conformations_to_test_dataset(dataset):
    test_datas = []
    for data in dataset:
        datas_ = []
        name = data.name[0]
        name = name.replace("->","")
        name = name.replace("(","")
        name = name.replace(")","")
        n_atoms = data.x.size(0)

        conf_path = "conformations//" + name + ".pickle"
        k = 0
        if os.path.exists(conf_path):
                with open(conf_path, 'rb') as handle:
                        distencies = pickle.load(handle)
                for j, dis_ in enumerate(distencies):
                        d = data.clone()
                        
                        pos = torch.from_numpy(dis_).float()
                        d.pos = pos
                        edge_index, edge_attr = get_cut_off_graph(d.edge_index, d.edge_attr, d.pos ,cut_off=6.)
                        d.edge_index = edge_index
                        d.edge_attr = edge_attr
                        datas_.append(d)
                        k = k + 1

        else:
                continue
        c_mask = data.x[:,0] == 2.0
        nmr_mask = data.x[:,-1] > -0.5
        mask = nmr_mask.logical_and(c_mask)
        test_datas.append( {"name":data.name,"n_atoms": n_atoms, "n_nmr":torch.sum(mask).item(), "n_mols": k,
                                        "loader" : DataLoader(datas_, batch_size = len(datas_), shuffle=False )})
    return test_datas

def main(path_to_mono, path_to_di, path_to_tri, fold):
    np.random.seed(0)
    torch.manual_seed(0)

    mols_mono = []
    with Chem.SDMolSupplier(path_to_mono) as suppl:
        for mol in suppl:
            mols_mono.append(mol)
    if not os.path.exists("roots/carbon/mono"):  
        os.makedirs("roots/carbon/mono")
    if not os.path.exists("roots/hydrogen/mono"):  
        os.makedirs("roots/hydrogen/mono")              
    mols_mono_carbon_d = list(Carbons13C("roots/carbon/mono", mols_mono))
    mols_mono_hydrogen_d = list(Hydrogens1H("roots/hydrogen/mono", mols_mono))

    mols_di = []
    with Chem.SDMolSupplier(path_to_di) as suppl:
        for mol in suppl:
            mols_di.append(mol)
    if not os.path.exists("roots/carbon/di"):  
        os.makedirs("roots/carbon/di") 
    if not os.path.exists("roots/hydrogen/di"):  
        os.makedirs("roots/hydrogen/di")             
    mols_di_carbon_d = list(Carbons13C("roots/carbon/di", mols_di))
    mols_di_hydrogen_d = list(Hydrogens1H("roots/hydrogen/di", mols_di))

    mols_tri = []
    with Chem.SDMolSupplier(path_to_tri) as suppl:
        for mol in suppl:
            mols_tri.append(mol)
    if not os.path.exists("roots/carbon/tri"):  
        os.makedirs("roots/carbon/tri")   
    if not os.path.exists("roots/hydrogen/tri"):  
        os.makedirs("roots/hydrogen/tri")             
    mols_tri_carbon_d = list(Carbons13C("roots/carbon/tri",mols_tri ))
    mols_tri_hydrogen_d = list(Hydrogens1H("roots/hydrogen/tri",mols_tri ))

    if not os.path.exists("conformations"):  
        os.makedirs("conformations") 
    create_conformer(mols_mono)
    create_conformer(mols_di,1)
    create_conformer(mols_tri,1)

    train_data_carbon_mo, test_data_carbon_mo = train_test_split(mols_mono_carbon_d, fold)
    train_data_carbon_di, test_data_carbon_di = train_test_split(mols_di_carbon_d, fold)
    train_data_carbon_tri, test_data_carbon_tri = train_test_split(mols_tri_carbon_d, fold)

    train_data_hydrogen_mo, test_data_hydrogen_mo = train_test_split(mols_mono_hydrogen_d, fold)
    train_data_hydrogen_di, test_data_hydrogen_di = train_test_split(mols_di_hydrogen_d, fold)
    train_data_hydrogen_tri, test_data_hydrogen_tri = train_test_split(mols_tri_hydrogen_d, fold)


    train_data_carbon = train_data_carbon_mo + train_data_carbon_di + train_data_carbon_tri
    train_data_hydrogen = train_data_hydrogen_mo + train_data_hydrogen_di + train_data_hydrogen_tri

    train_data_carbon = add_conformations_to_train_dataset(train_data_carbon)
    train_data_hydrogen = add_conformations_to_train_dataset(train_data_hydrogen)
    
    if not os.path.exists("datasets"):  
        os.makedirs("datasets") 
    
    with open("datasets/train_data_13C.pickle", 'wb') as handle:
        pickle.dump(train_data_carbon, handle)
    with open("datasets/train_data_1H.pickle", 'wb') as handle:
        pickle.dump(train_data_hydrogen, handle)

    test_data_carbon_mo = add_conformations_to_test_dataset(test_data_carbon_mo)
    test_data_carbon_di = add_conformations_to_test_dataset(test_data_carbon_di)
    test_data_carbon_tri = add_conformations_to_test_dataset(test_data_carbon_tri)

    test_data_hydrogen_mo = add_conformations_to_test_dataset(test_data_hydrogen_mo)
    test_data_hydrogen_di = add_conformations_to_test_dataset(test_data_hydrogen_di)
    test_data_hydrogen_tri = add_conformations_to_test_dataset(test_data_hydrogen_tri)

    with open("datasets/test_data_13C_mo.pickle", 'wb') as handle:
        pickle.dump(test_data_carbon_mo, handle)
    with open("datasets/test_data_13C_di.pickle", 'wb') as handle:
        pickle.dump(test_data_carbon_di, handle)
    with open("datasets/test_data_13C_tri.pickle", 'wb') as handle:
        pickle.dump(test_data_carbon_tri, handle)
    with open("datasets/test_data_1H_mo.pickle", 'wb') as handle:
        pickle.dump(test_data_hydrogen_mo, handle)
    with open("datasets/test_data_1H_di.pickle", 'wb') as handle:
        pickle.dump(test_data_hydrogen_di, handle)
    with open("datasets/test_data_1H_tri.pickle", 'wb') as handle:
        pickle.dump(test_data_hydrogen_tri, handle)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--data_path_mo', type=str, help='Path to train dataset mono')
    parser.add_argument('--data_path_di', type=str, help='Path to train dataset di')
    parser.add_argument('--data_path_tri', type=str, help='Path to train dataset tri')
    parser.add_argument('--fold', type=int, default=1, help='Fold in k-fold')

    args = parser.parse_args()
    main(args.data_path_mo, args.data_path_di, args.data_path_tri, args.fold)