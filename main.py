import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from model.model import O3Transformer
from model.norms import EquivariantLayerNorm
from e3nn import o3
import pickle

test_criterion = torch.nn.L1Loss()

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data_carb: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        super().__init__()
        self.gpu_id = 0
        self.model = model.to(self.gpu_id)
        self.train_data_carb = train_data_carb
        self.optimizer = optimizer
        self.derive_mean_and_std()
    def _run_batch(self, data: Data, criterion):
        self.optimizer.zero_grad()
        

        nmr_true =  data.x[:,-1]
        c_mask = data.x[:,0] == 2.0
        nmr_mask = data.x[:,-1] > -0.5
        mask = nmr_mask.logical_and(c_mask)    
        nmr_true =  data.x[:,-1]
        nmr_masked = (nmr_true[mask]- self.mean)/self.std
        out = self.model(x = data.x[:,0:2].long(), pos = data.pos.float(), 
                    edge_index = data.edge_index, edge_attr = 
                    data.edge_attr.long(), batch = data.batch)
        loss = criterion(out[mask].flatten(), nmr_masked.clone())
        loss.backward()
        
        self.optimizer.step()
        return loss.item()

    def _val_batch(self, data: Data, criterion):
        self.optimizer.zero_grad()
        nmr_true =  data.x[:,-1]
        c_mask = data.x[:,0] == 2.0
        nmr_mask = data.x[:,-1] > -0.5
        mask = nmr_mask.logical_and(c_mask)    
        nmr_true =  data.x[:,-1]
        nmr_masked = nmr_true[mask]
        out = self.model(x = data.x[:,0:2].long(), pos = data.pos.float(), 
                    edge_index = data.edge_index, edge_attr = 
                    data.edge_attr.long(), batch = data.batch)
        
        out_masked = out[mask]* self.std +  self.mean
        loss = test_criterion(out_masked.flatten(), nmr_masked)
        
        return loss.item()

    def _run_epoch(self, epoch: int,criterion):
        losses = []
        for i, data in enumerate(self.train_data_carb):                   
            torch.cuda.empty_cache()
            data = data.to(self.gpu_id)
            losses.append(self._run_batch(data, criterion))
            if (self.gpu_id == 0) and (i%50 ==0):
                self._save_checkpoint(epoch)
        return({sum(losses) / len(losses)})

    def _save_checkpoint(self, epoch: int):
        ckp = self.model.state_dict()
        PATH = "_checkpoint_epoch_" + str(epoch)+".pkl"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def derive_mean_and_std(self):
        nmrs = []
        for data in self.train_data_carb:
            nmr_true =  data.x[:,-1]
            c_mask = data.x[:,0] == 2.0
            nmr_mask = data.x[:,-1] > -0.5
            mask = nmr_mask.logical_and(c_mask)    
            nmrs.append(nmr_true[mask])
        nmrs = torch.cat(nmrs)
        self.mean = nmrs.mean()
        self.std = nmrs.std()

    def train(self, max_epochs: int, criterion, lr_reduce):
        
        for epoch in range(max_epochs):
            loss = self._run_epoch(epoch, criterion)
            print(f"Train loss {loss}")
            self._save_checkpoint(epoch)
            self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']*lr_reduce
                              

                
    def test(self, test_data: Data, message: str):
        torch.cuda.empty_cache()
        criterion = torch.nn.L1Loss()
    
        nmr_trues = []
        nmr_preds = []
        self.model.eval()
        with torch.no_grad():
            for i, loader_dict in enumerate(test_data):
                    N = loader_dict["n_mols"]
                    N_nmr = loader_dict["n_nmr"]
                    for data in loader_dict["loader"]:
                            data = data.to(self.gpu_id) 
                            c_mask = data.x[:,0] == 2.0
                            nmr_mask = data.x[:,-1] > -0.5
                            mask = nmr_mask.logical_and(c_mask) 
                            nmr_true =  data.x[:,-1]
                            nmr_masked = nmr_true[mask]
                            out = self.model(x = data.x[:,0:2].long(), pos = data.pos.float(), 
                                        edge_index = data.edge_index, edge_attr = data.edge_attr.long(), batch = data.batch)
                            out_masked = out[mask]*self.std + self.mean
                            
                            out = (out_masked.reshape(N,N_nmr).T).mean(dim = 1)
                            trues = nmr_masked[:N_nmr].detach().flatten()
                            nmr_trues.append(trues)
                            nmr_preds.append(out.detach().flatten())
            if nmr_trues:
                    nmr_trues_ = torch.cat(nmr_trues)
                    nmr_preds_ = torch.cat(nmr_preds)
    
                    l_lest = criterion(nmr_trues_.flatten(), nmr_preds_.flatten()).item()
                    print("Test error is " + str(l_lest) + " for " + message + ".")
        self.model.train()
                


def load_train_data(train_path: str):      

    with open(train_path, 'rb') as handle:
        train_data_carb = pickle.load(handle)

    train_data_carb_ = InMemoryDataset()       
    train_data_carb_.data, train_data_carb_.slices = train_data_carb_.collate(train_data_carb)    
    
    return train_data_carb_     
             

def load_train_test_objs(train_path:str, test_path_mo:str, test_path_di:str, test_path_tri:str):


    train_data_carb  = load_train_data(train_path)     
    with open(test_path_mo, 'rb') as handle:
        test_data_mo = pickle.load(handle)    
    with open(test_path_di, 'rb') as handle:
        test_data_di = pickle.load(handle)       
    with open(test_path_tri, 'rb') as handle:
        test_data_tri = pickle.load(handle) 

    model = O3Transformer(norm = EquivariantLayerNorm, n_input = 128,n_node_attr = 128,n_output =128, 
                                      irreps_hidden = o3.Irreps("64x0e + 32x1o + 8x2e"),n_layers = 7)
                                      
    return train_data_carb, test_data_mo, test_data_di, test_data_tri, model


def prepare_dataloader(dataset: Dataset, batch_size: int):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )

def prepare_test_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False)

def main(epochs: int, batch_size: int, train_path: str, test_path_mo: str, test_path_di: str, test_path_tri: str):

    train_carb, test_data_mo, test_data_di, test_data_tri, model = load_train_test_objs(train_path, test_path_mo, test_path_di, test_path_tri)

    train_data_carb = prepare_dataloader(train_carb, batch_size)   
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    trainer = Trainer(model,train_data_carb, optimizer)
    
    
    criterion = torch.nn.L1Loss()
    trainer.train(epochs, criterion, 0.1)
    trainer.test(test_data_mo,"mono")
    trainer.test(test_data_di,"di")
    trainer.test(test_data_tri, "tri")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--train_path', type=str, help='Path to train dataset')
    parser.add_argument('--test_path_mo', type=str, help='Path to test dataset')
    parser.add_argument('--test_path_di', type=str, help='Path to test dataset')
    parser.add_argument('--test_path_tri', type=str, help='Path to test dataset')

    parser.add_argument('--batch_size', default=32, type=int, help='Path to test dataset')
    parser.add_argument('--epochs', default=3, type=int, help='Path to test dataset')
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.train_path, args.test_path_mo, args.test_path_di, args.test_path_tri)
