#################################################################
# Defines the neural network model structure, including GNN and #
# other learning architectures. Part of this module reuses code #
# from the Symbolic Graph Neural Network.                       #
# Version: Residual Learning                                    #
#################################################################

import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Softplus, Sigmoid
from torch.autograd import Variable, grad
from torch_geometric.nn.aggr import PowerMeanAggregation  
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, MaxAggregation
import config

def get_edge_index(Ny, Nx):
    """
    Constructs edge indices for a 2D grid (4-connectivity).
    Returns:
    - edge_index: torch.LongTensor, shape (2, num_edges)
    """
    edges = []

    # Map each node to row-major ID: node_id = i * W + j
    for i in range(Ny):
        for j in range(Nx):
            node_id = i * Nx + j
            # Up
            if i > 0:
                edges.append((node_id, (i - 1) * Nx + j))
            # Down
            if i < Ny - 1:
                edges.append((node_id, (i + 1) * Nx + j))
            # Left
            if j > 0:
                edges.append((node_id, i * Nx + (j - 1)))
            # Right
            if j < Nx - 1:
                edges.append((node_id, i * Nx + (j + 1)))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index

class GN(MessagePassing):
    def __init__(self, n_f, msg_dim, out_dim, hidden=300, aggr='add'):
        super(GN, self).__init__(aggr=None)  # Set to None to override with custom aggregate
        self.aggr_str = aggr  # Store aggregation type string
        # self.p = config.p  # p-value for power-mean, default 3 is suitable for shocks
        self.out_dim = out_dim
        
        # Set aggregation objects based on aggr type
        if aggr == 'power_mean':
            self.aggr_obj = None  # Custom implementation, not using built-in
        elif aggr == 'custom':
            self.aggr_obj = None
        elif aggr == 'add' or aggr == 'sum':
            self.aggr_obj = SumAggregation()
        elif aggr == 'mean':
            self.aggr_obj = MeanAggregation()
        elif aggr == 'max':
            self.aggr_obj = MaxAggregation()
        else:
            raise ValueError(f"Unsupported aggr: {aggr}. Use 'add', 'mean', or 'power_mean'.")

        # Message function generator
        self.msg_fnc = Seq(
            # Input dim: 2*n_f, as message function receives concatenated features of two nodes
            Lin(2*n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            ## (Can turn on or off this layer:)
            # Lin(hidden, hidden), 
            # ReLU(),
            Lin(hidden, msg_dim)
        )
        
        # Node update function (after message aggregation)
        self.node_fnc = Seq(
            # Input: Concatenation of original node features (n_f) and aggregated messages (msg_dim)
            Lin(msg_dim + n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, out_dim)
        )

        # Added: Custom MLP aggregation (2 layers: Lin-ReLU-Lin)
        if aggr == 'custom':
            self.custom_mlp = Seq(
                Lin(msg_dim, hidden),  # msg_dim -> hidden
                ReLU(),
                Lin(hidden, msg_dim)   # hidden -> msg_dim (consistent output shape)
            )
    
    def forward(self, x, edge_index):
        # x: [n, n_f], node feature tensor (n nodes, n_f features per node)
        # edge_index: Edge index tensor, shape (2, num_edges)
        
        # Propagate calls message(), then aggregates based on aggr, and finally calls update()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
      
    def message(self, x_i, x_j):
        # x_i: [n_e, n_f]; x_j: [n_e, n_f]
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp: [E, 2 * in_channels]
        tmp = tmp.float()
        return self.msg_fnc(tmp)
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        if self.aggr_str == 'power_mean':
            # Custom power-mean (p=1.5), improved sign recovery to prevent NaN
            num_nodes = dim_size or (index.max().item() + 1)
            device = inputs.device
            
            # Step 1: |inputs|^p (Prevent NaN: clamp & epsilon)
            abs_inputs = torch.abs(inputs).clamp(min=1e-8)
            powered = abs_inputs.pow(self.p)
            powered = torch.clamp(powered, max=1e6)
            
            # Step 2: signed_powered = sign * |m|^p
            signed_powered = torch.sign(inputs) * powered
            
            # Step 3: scatter sum (supports ptr for batching)
            summed_powered = torch.zeros(num_nodes, inputs.size(1), device=device)
            summed_signed = torch.zeros(num_nodes, inputs.size(1), device=device)
            index_expanded = index.unsqueeze(1).expand_as(inputs)
            summed_powered.scatter_add_(0, index_expanded, powered)
            summed_signed.scatter_add_(0, index_expanded, signed_powered)
            
            # Step 4: count & mean
            count = torch.bincount(index, minlength=num_nodes).to(device).unsqueeze(1).expand_as(summed_powered)
            count = count.clamp(min=1)
            mean_powered = summed_powered / count
            mean_signed_powered = summed_signed / count
            
            # Step 5: Magnitude & Sign recovery
            amplitude = mean_powered.pow(1 / self.p)
            signs = torch.sign(mean_signed_powered)
            aggr_out = signs * amplitude
            
            return aggr_out
            
        elif self.aggr_str == 'custom':
            # Initial scatter (using mean as baseline, [N, msg_dim])
            num_nodes = dim_size or (index.max().item() + 1)
            device = inputs.device
            
            # Preliminary aggregation: mean
            summed = torch.zeros(num_nodes, inputs.size(1), device=device)
            index_expanded = index.unsqueeze(1).expand_as(inputs)
            summed.scatter_add_(0, index_expanded, inputs)
            
            count = torch.bincount(index, minlength=num_nodes).to(device).unsqueeze(1).expand_as(summed)
            count = count.clamp(min=1)
            temp_aggr = summed / count  # [N, msg_dim]
            
            # Fusion via 2-layer MLP (nonlinear enhancement of neighborhood info)
            aggr_out = self.custom_mlp(temp_aggr)
            aggr_out = torch.clamp(aggr_out, -1e3, 1e3)  # Prevent NaN
            
            return aggr_out
        else:
            # Fallback to built-in aggregation: call aggr_obj directly
            if self.aggr_obj is not None:
                return self.aggr_obj(inputs, index, ptr, dim_size)
            else:
                raise ValueError(f"No valid aggr_obj for {self.aggr_str}")
    
    def update(self, aggr_out, x=None):
        # aggr_out: [n, msg_dim]
        tmp = torch.cat([x, aggr_out], dim=1)
        tmp = tmp.float()
        residual = self.node_fnc(tmp)
        return residual
        # Note: In partial residual learning, one might only update specific dimensions 
        # (e.g., x_prev, y_prev) while keeping others constant.
    
# Graph Neural Network for PDE systems
class PGN(GN):
    def __init__(
        self, n_f, msg_dim, ndim, out_dim,
        edge_index, aggr='add', hidden=300):

        super(PGN, self).__init__(n_f, msg_dim, out_dim, hidden=hidden, aggr=aggr)
        self.edge_index = edge_index
        self.out_dim = out_dim # Output dimension
        self.ndim = ndim
    
    # Given the current graph structure and node features, calculate the feature 
    # derivatives (i.e., the dynamic rate of change) for each node.
    def just_derivative(self, g):
        # x: [n, n_f]
        x = g.x
        edge_index = g.edge_index
        
        return self.propagate(
                edge_index, 
                size=(x.size(0), x.size(0)),
                x=x)
    
    def loss(self, g, augment=True, square=False, augmentation=3, **kwargs):
        # Controls whether the loss function uses Mean Squared Error (L2) or Absolute Error (L1)
        if square:
            return torch.sum((g.y - self.just_derivative(g, augment=augment, augmentation=augmentation))**2)
        else:
            return torch.sum(torch.abs(g.y - self.just_derivative(g, augment=augment)))