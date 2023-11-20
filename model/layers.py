from e3nn import o3
import torch
from torch_scatter import scatter
from e3nn.math import soft_one_hot_linspace, soft_unit_step
import e3nn.nn as nn
from e3nn.util.jit import compile_mode

from .tensor_product_rescale import FullyConnectedTensorProductRescaleSwishGate
from .tensor_product_rescale import FullyConnectedTensorProductRescale
import torch.nn.functional as F

_RESCALE = True

class ComposeNetworkNorm(torch.nn.Module):
    def __init__(self,
                 network,
                 norm):
        super().__init__()

        self.network = network
        self.norm = norm
    def forward(self, node_input, node_attr, batch = None):
        node_output = self.network(node_input, node_attr)
        if batch is not None:
            return self.norm(node_output, batch)
        else:
            return self.norm(node_output)

@compile_mode('script')
class FeedForwardNetwork(torch.nn.Module):

    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_node_output, irreps_mlp_mid=None):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        
        self.fctp_1 = FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_mlp_mid, 
            bias=True, rescale=_RESCALE)
        self.fctp_2 = FullyConnectedTensorProductRescale(
            self.irreps_mlp_mid, self.irreps_node_attr, self.irreps_node_output, 
            bias=True, rescale=_RESCALE)
        
            
        
    def forward(self, node_input, node_attr, res = False, **kwargs):
        node_output = self.fctp_1(node_input, node_attr)
        node_output = self.fctp_2(node_output, node_attr)
        return node_output + node_input
    


class CompuseTransformerNorm_bond_attr(torch.nn.Module):
    def __init__(self, transformer, norm):
        super().__init__()
        self.transformer = transformer
        self.norm = norm

    def forward(self, node_input, pos, node_attr,edge_index, edge_attr, batch = None, norm_batch = None):
        node_input = self.transformer( node_input, pos, node_attr,edge_index, edge_attr, batch)
        if norm_batch is not None:
            return(self.norm(node_input, norm_batch))
        else:
            return(self.norm(node_input))



class TransformerLayer_with_bond(torch.nn.Module):
    def __init__(
        self,
        irreps_input,
        irreps_node_attr,
        irreps_output,
        irreps_query,
        irreps_key,
        edge_attr_dim,
        number_of_basis = 8
    ) -> None:
        super().__init__()
        self.irreps_sh = o3.Irreps.spherical_harmonics(2)
        self.number_of_basis = number_of_basis

        self.h_q = o3.Linear(irreps_input, irreps_query)
        self.self_interaction = o3.FullyConnectedTensorProduct(irreps_input, irreps_node_attr, irreps_output)
        self.tp_k = o3.FullyConnectedTensorProduct(irreps_input, self.irreps_sh, irreps_key, shared_weights=False)
        self.fc_k = nn.FullyConnectedNet([number_of_basis + edge_attr_dim, 128, self.tp_k.weight_numel], act=torch.nn.functional.silu)

        self.tp_v = o3.FullyConnectedTensorProduct(irreps_input, self.irreps_sh, irreps_output, shared_weights=False)
        self.fc_v = nn.FullyConnectedNet([number_of_basis + edge_attr_dim, 128, self.tp_v.weight_numel], act=torch.nn.functional.silu)
        self.dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "0e")
        self.max_radius = 6.0
    def forward(self,x, pos, node_attr,edge_index, edge_attr, batch):
    
        edge_src, edge_dst = edge_index#radius_graph(pos, self.max_radius, batch=batch)
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_length = edge_vec.norm(dim=1)

        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='smooth_finite',
            cutoff=True
        )
        edge_length_embedded = edge_length_embedded.mul(self.number_of_basis**0.5)

        edge_data = torch.cat([edge_length_embedded, edge_attr], dim = 1)
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / self.max_radius))

        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, True, normalization='component')

        q = self.h_q(x)
        k = self.tp_k(x[edge_src], edge_sh, self.fc_k(edge_data))
        v = self.tp_v(x[edge_src], edge_sh, self.fc_v(edge_data))
        
        temp_ = self.dot(q[edge_dst], k)
        exp = torch.exp(edge_weight_cutoff[:, None] * temp_)
        z = scatter(exp, edge_dst, dim=0, dim_size=len(x))
        z[z == 0] = 1
        alpha = exp / z[edge_dst]
        a = torch.sqrt(F.relu(alpha) +1e-14)#alpha.relu().sqrt() 
        return self.self_interaction(x, node_attr) + scatter(a * v, edge_dst, dim=0, dim_size=len(x))
        

class TransformerLayer_with_bond_invariant(torch.nn.Module):
    def __init__(
        self,
        irreps_input,
        irreps_node_attr,
        irreps_output,
        irreps_query,
        irreps_key,
        edge_attr_dim,
        number_of_basis = 8
    ) -> None:
        super().__init__()
        self.irreps_sh = o3.Irreps.spherical_harmonics(0)
        self.number_of_basis = number_of_basis
        self.h_q = o3.Linear(irreps_input, irreps_query)
        self.self_interaction = o3.FullyConnectedTensorProduct(irreps_input, irreps_node_attr, irreps_output)
        self.tp_k = o3.FullyConnectedTensorProduct(irreps_input, self.irreps_sh, irreps_key, shared_weights=False)
        self.fc_k = nn.FullyConnectedNet([number_of_basis + edge_attr_dim, 128, self.tp_k.weight_numel], act=torch.nn.functional.silu)

        self.tp_v = o3.FullyConnectedTensorProduct(irreps_input, self.irreps_sh, irreps_output, shared_weights=False)
        self.fc_v = nn.FullyConnectedNet([number_of_basis + edge_attr_dim, 128, self.tp_v.weight_numel], act=torch.nn.functional.silu)
        self.dot = o3.FullyConnectedTensorProduct(irreps_query, irreps_key, "0e")
        self.max_radius = 6.0
    def forward(self,x, pos, node_attr,edge_index, edge_attr, batch):
    
        edge_src, edge_dst = edge_index#radius_graph(pos, self.max_radius, batch=batch)
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_length = edge_vec.norm(dim=1)

        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='smooth_finite',
            cutoff=True
        )
        edge_length_embedded = edge_length_embedded.mul(self.number_of_basis**0.5)

        edge_data = torch.cat([edge_length_embedded, edge_attr], dim = 1)
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / self.max_radius))

        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, True, normalization='component')

        q = self.h_q(x)
        k = self.tp_k(x[edge_src], edge_sh, self.fc_k(edge_data))
        v = self.tp_v(x[edge_src], edge_sh, self.fc_v(edge_data))
        temp_ = self.dot(q[edge_dst], k)
        exp = torch.exp(edge_weight_cutoff[:, None] * temp_)
        z = scatter(exp, edge_dst, dim=0, dim_size=len(x))
        z[z == 0] = 1
        alpha = exp / z[edge_dst]
        a = torch.sqrt(F.relu(alpha) +1e-14)#alpha.relu().sqrt() 
        return self.self_interaction(x, node_attr) + scatter(a * v, edge_dst, dim=0, dim_size=len(x))
