from e3nn import o3
import torch
from.layers import TransformerLayer_with_bond, CompuseTransformerNorm_bond_attr, FeedForwardNetwork, ComposeNetworkNorm, TransformerLayer_with_bond_invariant
from.norms import EquivariantLayerNorm

class O3Transformer(torch.nn.Module):
    def __init__(
        self,
        n_input = 32,
        n_node_attr = 32,
        n_output =32,
        n_edge_attr = 32,
        irreps_query = o3.Irreps("32x0e + 16x1o + 8x2e"),
        irreps_key = o3.Irreps("32x0e + 16x1o + 8x2e"),
        irreps_hidden = o3.Irreps("32x0e + 16x1o + 8x2e"),
        number_of_basis = 16,
        n_layers = 6,
        norm = EquivariantLayerNorm
    ) -> None:
        super().__init__()
        input_irreps = (o3.Irreps("0e")*n_input*2).simplify()
        node_attr_irreps = (o3.Irreps("0e")*n_node_attr*2).simplify()
        irreps_output = (o3.Irreps("0e")*n_output).simplify()


        self.input_embedding = torch.nn.Embedding(10,n_input)
        self.input_embedding_hydrogen = torch.nn.Embedding(5,n_input)

        self.attr_embedding = torch.nn.Embedding(10,n_node_attr)
        self.attr_embedding_h = torch.nn.Embedding(5,n_node_attr)

        self.edge_attr_embedding = torch.nn.Embedding(6,n_edge_attr)

        layers = []
        feed_forward_layers = []
        layers.append(CompuseTransformerNorm_bond_attr(TransformerLayer_with_bond(irreps_input = input_irreps,
                                                    irreps_node_attr = node_attr_irreps,
                                                    irreps_output = irreps_hidden,
                                                    irreps_query = irreps_query,
                                                    irreps_key = irreps_key,
                                                    number_of_basis = number_of_basis,
                                                    edge_attr_dim = n_edge_attr),
                                                    norm(irreps=irreps_hidden)))
        feed_forward_layers.append(ComposeNetworkNorm(FeedForwardNetwork(irreps_hidden,node_attr_irreps, irreps_hidden),
                           norm(irreps=irreps_hidden)))
        for i in range(n_layers-2):
            layers.append(CompuseTransformerNorm_bond_attr(TransformerLayer_with_bond(irreps_input = irreps_hidden,
                                                        irreps_node_attr = node_attr_irreps,
                                                        irreps_output = irreps_hidden,
                                                        irreps_query = irreps_query,
                                                        irreps_key = irreps_key,
                                                        number_of_basis = number_of_basis,
                                                        edge_attr_dim = n_edge_attr),
                                                        norm(irreps=irreps_hidden)))   
            feed_forward_layers.append(ComposeNetworkNorm(FeedForwardNetwork(irreps_hidden,node_attr_irreps, irreps_hidden),
                    norm(irreps=irreps_hidden)))
        layers.append(CompuseTransformerNorm_bond_attr(TransformerLayer_with_bond(irreps_input = irreps_hidden,
                                                                irreps_node_attr = node_attr_irreps,
                                                                irreps_output = irreps_output,
                                                                irreps_query = irreps_query,
                                                                irreps_key = irreps_key,
                                                                number_of_basis = number_of_basis,
                                                                edge_attr_dim = n_edge_attr),
                                                                norm(irreps=irreps_output)))  
        self.layers = torch.nn.ModuleList(layers)     
        self.feed_forward_layers = torch.nn.ModuleList(feed_forward_layers)

        self.output_mlp = torch.nn.Sequential(torch.nn.Linear(n_output, n_output*3), torch.nn.ELU(), torch.nn.Linear(n_output*3, 1))

    def forward(self, x, pos, edge_index, edge_attr, batch = None, norm_batch = None):

        node_attr1 = self.attr_embedding(x[:,0])
        node_attr2 = self.attr_embedding_h(x[:,1])
        node_attr = torch.cat([node_attr1, node_attr2], dim = 1)
        x1 = self.input_embedding(x[:,0])
        x2 = self.input_embedding_hydrogen(x[:,1])
        x = torch.cat( [x1, x2], dim = 1)
        edge_attr = self.edge_attr_embedding(edge_attr.long())

        for i, layer in enumerate(self.layers):
            x = layer(x,pos,node_attr, edge_index, edge_attr, batch, norm_batch)
            if i < len(self.layers) -1:
                x = self.feed_forward_layers[i](x, node_attr, norm_batch)
        
        x = self.output_mlp(x)
        return x


class O3Transformer_invariant(torch.nn.Module):
    def __init__(
        self,
        n_input = 32,
        n_node_attr = 32,
        n_output =32,
        n_edge_attr = 32,
        irreps_query = o3.Irreps("64x0e"),
        irreps_key = o3.Irreps("64x0e"),
        irreps_hidden = o3.Irreps("64x0e"),
        number_of_basis = 16,
        n_layers = 6,
        norm = EquivariantLayerNorm
    ) -> None:
        super().__init__()
        input_irreps = (o3.Irreps("0e")*n_input*2).simplify()
        node_attr_irreps = (o3.Irreps("0e")*n_node_attr*2).simplify()
        irreps_output = (o3.Irreps("0e")*n_output).simplify()


        self.input_embedding = torch.nn.Embedding(10,n_input)
        self.input_embedding_hydrogen = torch.nn.Embedding(5,n_input)

        self.attr_embedding = torch.nn.Embedding(10,n_node_attr)
        self.attr_embedding_h = torch.nn.Embedding(5,n_node_attr)

        self.edge_attr_embedding = torch.nn.Embedding(6,n_edge_attr)

        layers = []
        feed_forward_layers = []
        layers.append(CompuseTransformerNorm_bond_attr(TransformerLayer_with_bond_invariant(irreps_input = input_irreps,
                                                    irreps_node_attr = node_attr_irreps,
                                                    irreps_output = irreps_hidden,
                                                    irreps_query = irreps_query,
                                                    irreps_key = irreps_key,
                                                    number_of_basis = number_of_basis,
                                                    edge_attr_dim = n_edge_attr),
                                                    norm(irreps=irreps_hidden)))
        feed_forward_layers.append(ComposeNetworkNorm(FeedForwardNetwork(irreps_hidden,node_attr_irreps, irreps_hidden),
                           norm(irreps=irreps_hidden)))
        for i in range(n_layers-2):
            layers.append(CompuseTransformerNorm_bond_attr(TransformerLayer_with_bond_invariant(irreps_input = irreps_hidden,
                                                        irreps_node_attr = node_attr_irreps,
                                                        irreps_output = irreps_hidden,
                                                        irreps_query = irreps_query,
                                                        irreps_key = irreps_key,
                                                        number_of_basis = number_of_basis,
                                                        edge_attr_dim = n_edge_attr),
                                                        norm(irreps=irreps_hidden)))   
            feed_forward_layers.append(ComposeNetworkNorm(FeedForwardNetwork(irreps_hidden,node_attr_irreps, irreps_hidden),
                    norm(irreps=irreps_hidden)))
        layers.append(CompuseTransformerNorm_bond_attr(TransformerLayer_with_bond_invariant(irreps_input = irreps_hidden,
                                                                irreps_node_attr = node_attr_irreps,
                                                                irreps_output = irreps_output,
                                                                irreps_query = irreps_query,
                                                                irreps_key = irreps_key,
                                                                number_of_basis = number_of_basis,
                                                                edge_attr_dim = n_edge_attr),
                                                                norm(irreps=irreps_output)))  
        self.layers = torch.nn.ModuleList(layers)     
        self.feed_forward_layers = torch.nn.ModuleList(feed_forward_layers)

        self.output_mlp = torch.nn.Sequential(torch.nn.Linear(n_output, n_output*3), torch.nn.ELU(), torch.nn.Linear(n_output*3, 1))

    def forward(self, x, pos, edge_index, edge_attr, batch = None, norm_batch = None):

        node_attr1 = self.attr_embedding(x[:,0])
        node_attr2 = self.attr_embedding_h(x[:,1])
        node_attr = torch.cat([node_attr1, node_attr2], dim = 1)
        x1 = self.input_embedding(x[:,0])
        x2 = self.input_embedding_hydrogen(x[:,1])
        x = torch.cat( [x1, x2], dim = 1)
        edge_attr = self.edge_attr_embedding(edge_attr.long())

        for i, layer in enumerate(self.layers):
            x = layer(x,pos,node_attr, edge_index, edge_attr, batch, norm_batch)
            if i < len(self.layers) -1:
                x = self.feed_forward_layers[i](x, node_attr, norm_batch)
        
        x = self.output_mlp(x)
        return x


