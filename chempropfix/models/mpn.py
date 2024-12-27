from typing import List, Union, Tuple
from functools import reduce
from torch_geometric import nn as pnn
import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from chempropfix.args import TrainArgs
from chempropfix.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chempropfix.nn_utils import index_select_ND, get_activation_function
from torch_scatter import scatter_sum, scatter_mean
# from .polygnn import polygnn_mp
from torch_geometric.nn.norm.batch_norm import BatchNorm
import time
from torch_geometric.nn.models import MLP
#from onmt.modules.embeddings import PositionalEncoding
#from onmt.modules.position_ffn import PositionwiseFeedForward
import networkx as nx
import math
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn.models import GCN,GAT,GraphSAGE,GIN,BasicGNN
import torch.nn.functional as F
from .layers import AttnEncoderXL, AttnEncoderADJ, ffn
# from .graphformer import graphformerEncoder
from typing import Any, List, Optional, Tuple,Final
from torch import Tensor
from torch_geometric.utils import cumsum, to_dense_batch, scatter
from torch_geometric.typing import OptTensor
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGPooling
from typing import Callable, Optional, Union
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset,ones, zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm
from torch_geometric.nn.dense import DenseGINConv
import torch.nn.functional as F
import pdb
import logging
import csv
from chempropfix.models.gcl import E_GCL,unsorted_segment_sum
from chempropfix.models.blocks import *
from e3nn import o3
from torch_geometric.nn.inits import glorot, zeros


logging.basicConfig(filename="/home/jiaopanyu/4-result/train/model.log", level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def visual_matrix(matrix,pth,idx,mtype=None,left=None,right=None):
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(matrix.cpu().detach().numpy(), cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    fig.colorbar(cax)
    if right:
        arr1 = np.arange(left,right)
        lens = np.concatenate((arr1, arr1))
        ax.set_xticks(np.arange(len(matrix[0])))
        ax.set_yticks(np.arange(len(matrix)))
        ax.set_xticklabels(lens, rotation=45)  # 旋转x轴标签
        ax.set_yticklabels(lens)
    else:
        ax.set_xticks(np.arange(len(matrix[0])))
        ax.set_yticks(np.arange(len(matrix)))
        ax.set_xticklabels(np.arange(len(matrix[0])), rotation=45)  # 旋转x轴标签
        ax.set_yticklabels(np.arange(len(matrix)))
    ax.grid(True, which='both', color='gray', linewidth=0.5, linestyle='--')
    plt.tight_layout()  # 自动调整布局
    if mtype:
        plt.savefig(f"{pth}/{idx}_{mtype}.png")
    else:
        plt.savefig(f"{pth}/{idx}.png")
    plt.close()

def to_dense_adj(
    edge_index: Tensor,
    batch: OptTensor = None,
    edge_attr: OptTensor = None,
    max_num_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tensor:

    if batch_size is None:
        batch_size = int(batch.max()) + 1 if batch.numel() > 0 else 1

    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='sum')
    cum_nodes = cumsum(num_nodes)

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]
    
    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    elif ((idx1.numel() > 0 and idx1.max() >= max_num_nodes)
          or (idx2.numel() > 0 and idx2.max() >= max_num_nodes)):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    A = torch.ones(idx0.numel(), device=edge_index.device)
    size = [batch_size, max_num_nodes, max_num_nodes]
    attr_size = size +  list(edge_attr.size())[1:]
    flattened_size = batch_size * max_num_nodes * max_num_nodes

    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    attr = scatter(edge_attr, idx, dim=0, dim_size=flattened_size, reduce='sum')
    adj = scatter(A, idx, dim=0, dim_size=flattened_size, reduce='sum')
    attr = attr.view(attr_size)
    adj = adj.view(size)

    return adj, attr

def repeat_interleave(
    repeats: List[int],
    device: Optional[torch.device] = None,
) -> Tensor:
    outs = [torch.full((n, ), i, device=device) for i, n in enumerate(repeats)]
    return torch.cat(outs, dim=0)

def coord2radial(edge_index, coord,norm_diff):
        row, col = edge_index   
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        if norm_diff:
            mean = torch.mean(radial)
            std = torch.std(radial)
            radial  = (radial - mean)/std
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)
        return radial, coord_diff

def compute_dirichlet_energy(x, edge_index):
    row, col = edge_index
    N = x.shape[1]
    diff = x[row] - x[col]
    diff_squared = diff.pow(2).sum(dim=1)
    dirichlet_energy = diff_squared.sum() / N
    return dirichlet_energy.mean()

def compute_mean_average_distance(x, edge_index):
    row, col = edge_index
    N = x.shape[1]
    diff = F.cosine_similarity(x[row].unsqueeze(0),x[col].unsqueeze(0))
    mad = (1-diff).sum() / N
    return mad.mean()

class POLYGINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" ß<https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self,in_channels,out_channels, nn: Callable, eps: float = 0., train_eps: bool = False, node_hidden: Optional[int] = None, edge_hidden: Optional[int] = None,activation:Callable=None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.edge_mapper = ffn(edge_hidden, edge_hidden,capacity=1,activation=activation)
        self.residual_mapping = ffn(in_channels+edge_hidden, in_channels,capacity=1,activation=activation)
        self.nn = nn
        self.initial_eps = eps
        self.layer_norm = pnn.norm.LayerNorm(in_channels,1e-7,mode='node')
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        self.reset_parameters() 
    
    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
       
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,edge_attr: Union[Tensor, OptPairTensor],
                size: Size = None,batch =None,batch_size = None) -> Tensor:
      
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = self.propagate(edge_index,size=size,x=x,edge_attr=edge_attr) 
        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r
        # out = self.layer_norm(out)
        out = self.nn(out)
        return out
    
    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        edge_attr = self.edge_mapper(edge_attr)
        x = self.residual_mapping(torch.concat((x_j,edge_attr),dim=-1)) 
        return x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

class POLYGIN(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool] = False

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels,in_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return POLYGINConv(in_channels,out_channels,mlp, **kwargs)
    
class EpicCoorEgnn(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels,in_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return EpicEgnnConv(in_channels,out_channels,mlp, **kwargs)

class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPNEncoder, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.gnn_dropout
        self.undirected = args.undirected
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm
        self.with_attn = args.with_attn
        if self.with_attn:
            self.attention_encoder = AttnEncoderXL(args,bond_fdim)

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        # layer after concatenating the descriptors if args.atom_descriptors == descriptors
        if args.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(self.hidden_size + self.atom_descriptors_size,
                                                    self.hidden_size + self.atom_descriptors_size,)
        self.mol_graph2data_layer = mol_graph2data(args)

    def forward(self,
                mol_graph: BatchMolGraph,
                atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atomic descriptors
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [np.zeros([1, atom_descriptors_batch[0].shape[1]])] + atom_descriptors_batch   # padding the first with 0 to match the atom_hiddens
            atom_descriptors_batch = torch.from_numpy(np.concatenate(atom_descriptors_batch, axis=0)).float().to(self.device)

        f_atoms, f_bonds, w_atoms, w_bonds, a2b, b2a, b2revb, \
        a_scope, b_scope, degree_of_polym, distances, node_paths, edge_paths,f_map,f_xyz= mol_graph.get_components(atom_messages=self.atom_messages)
        f_atoms, f_bonds, w_atoms, w_bonds, a2b, b2a, b2revb,f_map,f_xyz = f_atoms.to(self.device), f_bonds.to(self.device), \
                                                               w_atoms.to(self.device), w_bonds.to(self.device), \
                                                               a2b.to(self.device), b2a.to(self.device), \
                                                               b2revb.to(self.device),f_map.to(self.device), \
                                                               f_xyz.to(self.device)
        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        
        _, edge_index, _, _, _, _, _, _, _, _, _ = self.mol_graph2data_layer(mol_graph)
        # Message passing            
        for i in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2
            if self.args.visual == 21 and i == 1:
                _a,_b = compute_dirichlet_energy(message,edge_index),compute_mean_average_distance(message,edge_index)

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                nei_a_weight = index_select_ND(w_bonds, a2b)  # num_atoms x max_num_bonds
                # weight nei_a_message based on edge weights
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1) * weight(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = dot(nei_a_message,nei_a_weight)      rev_message
                nei_a_message = nei_a_message * nei_a_weight[..., None]  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                
                # For directed
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message * w_bonds[..., None]  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        # 取边的上层节点，并加权
        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        nei_a_weight = index_select_ND(w_bonds, a2x)  # num_atoms x max_num_bonds
        nei_a_message = nei_a_message * nei_a_weight[..., None]  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        if self.args.visual == 21:
            a_,b_ = compute_dirichlet_energy(atom_hiddens,edge_index),compute_mean_average_distance(atom_hiddens,edge_index)
            return torch.concat([(a_/_a).unsqueeze(0),(b_/_b).unsqueeze(0)],dim=0)
            
        # concatenate the atom descriptors
        if atom_descriptors_batch is not None:
            if len(atom_hiddens) != len(atom_descriptors_batch):
                raise ValueError(f'The number of atoms is different from the length of the extra atom features')

            atom_hiddens = torch.cat([atom_hiddens, atom_descriptors_batch], dim=1)     # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)                    # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.dropout_layer(atom_hiddens)                             # num_atoms x (hidden + descriptor size)
            
            
        # 实在改不动了，暂时这样写把
        atom_hiddens = atom_hiddens[1:]    
        x, edge_index, edge_attr, w_atoms, w_bonds, b2revb_individual, distances, node_paths, edge_paths, batch, ptr = self.mol_graph2data_layer(mol_graph)
        
        if self.with_attn:
            lengths = ptr[1:] - ptr[:-1]
            dense_batch, mask = to_dense_batch(torch.arange(atom_hiddens.size(0),device=self.device), batch=batch)
            mask = ~(mask.unsqueeze(2) & mask.unsqueeze(1))        # [batch_size, max_size] => [batch_size,max_size,max_size]
            node_count = torch.bincount(batch)
            node_features_split = torch.split(atom_hiddens, node_count.tolist())
            max_length = dense_batch.size(1)
            padded_tensor = torch.stack([torch.cat([feat, torch.zeros(max_length - feat.size(0), feat.size(1),device=self.device)]) for feat in node_features_split])

            atom_message, attns = self.attention_encoder(atom_hiddens,padded_tensor,mask,edge_index,edge_attr,ptr,batch,distances,node_paths,edge_paths)
            # atom_message = torch.cat([atom_message[i, :a,:] for i,a in enumerate(lengths)], dim=0)

            attns = torch.sum(attns,dim=1)
            
            if self.args.visual:
                return attns
                torch.save(attns, '/home/chenlidong/polyAttn/notebooks/tensor.pt')
            
            attns = attns.masked_fill(mask, 0)
            attns = torch.mean(attns,dim=1)
            atom_message = atom_message * attns.unsqueeze(-1)       # 需要再次加上w信息吗？
            poly_vec = torch.sum(atom_message,dim=1) / torch.sum(attns,dim=1).unsqueeze(1)
        else:
            # Readout
            poly_vec = []
            for i, (a_start, a_size) in enumerate(a_scope):
                if a_size == 0:
                    poly_vec.append(self.cached_zero_vector)
                else:
                    cur_hiddens = atom_hiddens.narrow(0, a_start-1, a_size)
                    mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                    w_atom_vec = w_atoms.narrow(0, a_start-1, a_size)
                    # if input are polymers, weight atoms from each repeating unit according to specified monomer fractions
                    # weight h by atom weights (weights are all 1 for non-polymer input)
                    mol_vec = w_atom_vec[..., None] * mol_vec
                    # weight each atoms at readout
                    if self.aggregation == 'mean':
                        mol_vec = mol_vec.sum(dim=0) / w_atom_vec.sum(dim=0)  # if not --polymer, w_atom_vec.sum == a_size
                    elif self.aggregation == 'sum':
                        mol_vec = mol_vec.sum(dim=0)
                    elif self.aggregation == 'norm':
                        mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm

                    # if input are polymers, multiply mol vectors by degree of polymerization
                    # if not --polymer, Xn is 1
                    mol_vec = degree_of_polym[i] * mol_vec

                    poly_vec.append(mol_vec)

            poly_vec = torch.stack(poly_vec, dim=0)  # (num_molecules, hidden_size)

        return poly_vec  # num_molecules x hidden

class mol_graph2data(nn.Module):
    def __init__(self,args):
        super(mol_graph2data,self).__init__()
        self.args = args

    def forward(self,
                mol_graph: BatchMolGraph,
                atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        f_atoms, f_bonds,a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        repeats = [i[1] for i in a_scope]  # 原子数
        batch = repeat_interleave(repeats, device=self.args.device)
        ptr = cumsum(torch.tensor(repeats, device=self.args.device))
        b2revb_individual = []
        edge_index = [[],[]]
        # 每一个分子中原子数和边数
        for idx, (a_scope_i, b_scope_i) in enumerate(zip(a_scope, b_scope)):
            b2a_i = b2a[b_scope_i[0]:b_scope_i[0] + b_scope_i[1]]
            b2revb_i = b2revb[b_scope_i[0]:b_scope_i[0] + b_scope_i[1]]
            edge_index[0].extend((b2a_i).tolist())
            edge_index[1].extend((b2a_i[b2revb_i - b_scope_i[0]]).tolist())
            # 如果使用pyG，那么就不需要b2revb_individual
            # b2revb_individual.append(b2revb[b_scope_i[0]:b_scope_i[0]+b_scope_i[1]]-b_scope_i[0])
            # b2revb_individual = torch.cat(b2revb_individual,dim=0)
        edge_index = torch.tensor(edge_index,dtype=torch.long,device=self.args.device)-1    #因为chemprop内部的设置，这里必须-1
        return f_atoms[1:,:].to(self.args.device),\
               edge_index, \
               f_bonds[1:,:].to(self.args.device),\
               b2revb_individual, \
               batch, ptr

class mol_graph2data_adj(nn.Module):
    def __init__(self,args):
        super(mol_graph2data_adj,self).__init__()
        self.args = args
        self.device =  args.device

    def forward(self,
        mol_graph: BatchMolGraph,
        atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        b2a_list = [b2a[start:start + length] for start, length in b_scope]
        b2revb_list = [b2revb[start:start + length] - start for start, length in b_scope]
        edge_index = torch.stack([torch.cat(b2a_list), torch.cat([b2a_i[b2revb_i] for b2a_i, b2revb_i in zip(b2a_list, b2revb_list)])])-1
        repeats = [i[1] for i in a_scope]
        max_num_nodes = max(repeats)
        batch = repeat_interleave(repeats, device=self.device)
        A, E = to_dense_adj(edge_index.to(self.device),batch,f_bonds[1:,:].to(self.device),max_num_nodes)
        X = f_atoms[1:,:].to(self.device)
        
        dense_batch, mask = to_dense_batch(torch.arange(X.size(0),device=self.device), batch=batch)
        flattened_dense_batch = dense_batch.view(-1)
        X =         torch.index_select(X,       0, flattened_dense_batch).view(-1, max_num_nodes, X.size(1) ).masked_fill(~mask.unsqueeze(-1),0)    # [batch_size * max_atom_size * hidden_size]
        mask = ~(mask.unsqueeze(2) & mask.unsqueeze(1))
        
        b = X.size(0)
    
        """
        i_max_len = mol_lens[:,0] + mol_lens[:,1]
        
        if self.args.mask_mode == 1:        # 全部放开
            pass
        elif self.args.mask_mode == 2:      # 左下右上mask
            for i in range(b):
                mask[i,0:mol_lens[i,0],mol_lens[i,0]:i_max_len[i]] = True                           # 右上
                mask[i,mol_lens[i,0]:i_max_len[i],0:mol_lens[i,0]] = True                           # 左下
                mask[i,0:mol_lens[i,0],0:mol_lens[i,0]] = False                                     # 左上
                mask[i,mol_lens[i,0]:i_max_len[i],mol_lens[i,0]:i_max_len[i]] = False               # 右下
        elif self.args.mask_mode == 3:      # 左上右下mask
            for i in range(b):
                mask[i,0:mol_lens[i,0],mol_lens[i,0]:i_max_len[i]] = False                          # 右上
                mask[i,mol_lens[i,0]:i_max_len[i],0:mol_lens[i,0]] = False                          # 左下
                mask[i,0:mol_lens[i,0],0:mol_lens[i,0]] = True                                      # 左上
                mask[i,mol_lens[i,0]:i_max_len[i],mol_lens[i,0]:i_max_len[i]] = True                # 右下
        elif self.args.mask_mode == 4:      # 左下右上mask,termini除外
            for i in range(b):
                mask[i,0:mol_lens[i,0],mol_lens[i,0]:i_max_len[i]] = True                           # 右上
                mask[i,mol_lens[i,0]:i_max_len[i],0:mol_lens[i,0]] = True                           # 左下
                mask[i,0:mol_lens[i,0],0:mol_lens[i,0]] = False                                     # 左上
                mask[i,mol_lens[i,0]:i_max_len[i],mol_lens[i,0]:i_max_len[i]] = False               # 右下
            index_a,index_b=torch.where(X[:,:,-1]==2.)  # 找到termini
            mask[index_a,index_b]=False
            mask[index_a,:,index_b]=False
            for i in range(b):
                mask[i,:,i_max_len[i]:]=True
                mask[i,i_max_len[i]:,:]=True
        elif self.args.mask_mode == 5:      # 左上右下mask,termini和邻居除外
            for i in range(b):
                mask[i,0:mol_lens[i,0],mol_lens[i,0]:i_max_len[i]] = True                           # 右上
                mask[i,mol_lens[i,0]:i_max_len[i],0:mol_lens[i,0]] = True                           # 左下
                mask[i,0:mol_lens[i,0],0:mol_lens[i,0]] = False                                     # 左上
                mask[i,mol_lens[i,0]:i_max_len[i],mol_lens[i,0]:i_max_len[i]] = False               # 右下
            index_a,index_b=torch.where(X[:,:,-1]==2.)  # 找到termini
            index_c,index_d = torch.where(A[index_a,index_b]==1) # 找到termini的邻居
            mask[index_a,index_b]=False
            mask[index_a,:,index_b]=False
            mask[index_a[index_c],index_d] = False
            mask[index_a[index_c],:,index_d] = False
            for i in range(b):
                mask[i,:,i_max_len[i]:]=True
                mask[i,i_max_len[i]:,:]=True
        elif self.args.mask_mode == 6:      # 只开放那8个点
           
            mask = True
            index_a,index_b=torch.where(X[:,:,-1]==2.)
            
        
        A[~mask & (A==0)] = 0.0001    # 这个很有必要！！！
        """
        return X,\
               E.to(self.device),\
               A.to(self.device), \
               mask,

class monomer_split(nn.Module):
    def __init__(self,node_hidden,edge_hidden,hidden_size):
        super().__init__()
        self.node_hidden = node_hidden
        self.edge_hidden = edge_hidden
        self.hidden_size = hidden_size
    
    def forward(self, X, E, A, w, mask, mol_lens):
        b = X.size(0)
        m, n = torch.max(mol_lens[:,0]), torch.max(mol_lens[:,1])
        # for 循环实现
        X1 = torch.zeros((b, 2 * m, self.hidden_size),device=X.device)
        X2 = torch.zeros((b, 2 * n, self.hidden_size),device=X.device)
        E1 = torch.zeros((b, 2 * m, 2 * m, self.edge_hidden),device=X.device)
        E2 = torch.zeros((b, 2 * n, 2 * n, self.edge_hidden),device=X.device)
        A1 = torch.zeros((b, 2 * m, 2 * m),device=X.device)
        A2 = torch.zeros((b, 2 * n, 2 * n),device=X.device)
        w1 = torch.zeros((b, 2 * m, 1),device=X.device)
        w2 = torch.zeros((b, 2 * n, 1),device=X.device)
        mask1 = torch.ones((b, 2 * m, 2 * m),device=X.device,dtype=torch.bool)
        mask2 = torch.ones((b, 2 * n, 2 * n),device=X.device,dtype=torch.bool)
        
        i_max_len = mol_lens[:,0] + mol_lens[:,1]
        for i in range(b):
            
            X1[i,:mol_lens[i,0]] = X[i,:mol_lens[i,0]]
            X1[i,mol_lens[i,0]:2*mol_lens[i,0]] = X[i,:mol_lens[i,0]]
            
            X2[i,:mol_lens[i,1]] = X[i,:mol_lens[i,1]]
            X2[i,mol_lens[i,1]:2*mol_lens[i,1]] = X[i,:mol_lens[i,1]]
            
            E1[i,:mol_lens[i,0],:mol_lens[i,0]] = E[i,:mol_lens[i,0],:mol_lens[i,0]]
            E1[i,mol_lens[i,0]:2*mol_lens[i,0],mol_lens[i,0]:2*mol_lens[i,0]] = E[i,:mol_lens[i,0],:mol_lens[i,0]]
            
            E2[i,:mol_lens[i,1],:mol_lens[i,1]] = E[i,mol_lens[i,0]:i_max_len[i],mol_lens[i,0]:i_max_len[i]]
            E2[i,mol_lens[i,1]:2*mol_lens[i,1],mol_lens[i,1]:2*mol_lens[i,1]] = E[i,mol_lens[i,0]:i_max_len[i],mol_lens[i,0]:i_max_len[i]]
            
            A1[i,:mol_lens[i,0],:mol_lens[i,0]] = A[i,:mol_lens[i,0],:mol_lens[i,0]]
            A1[i,mol_lens[i,0]:2*mol_lens[i,0],mol_lens[i,0]:2*mol_lens[i,0]] = A[i,:mol_lens[i,0],:mol_lens[i,0]]
            
            A2[i,:mol_lens[i,1],:mol_lens[i,1]] = A[i,mol_lens[i,0]:i_max_len[i],mol_lens[i,0]:i_max_len[i]]
            A2[i,mol_lens[i,1]:2*mol_lens[i,1],mol_lens[i,1]:2*mol_lens[i,1]] = A[i,mol_lens[i,0]:i_max_len[i],mol_lens[i,0]:i_max_len[i]]

            # w1[i,:mol_lens[i,0]] = w[i,:mol_lens[i,0]]
            # w1[i,mol_lens[i,0]:2*mol_lens[i,0]] = w[i,:mol_lens[i,0]]
            
            # w2[i,:mol_lens[i,1]] = w[i,:mol_lens[i,1]]
            # w2[i,mol_lens[i,1]:2*mol_lens[i,1]] = w[i,:mol_lens[i,1]]
            mask[i,0:mol_lens[i,0],mol_lens[i,0]:i_max_len[i]] = False
            mask[i,mol_lens[i,0]:i_max_len[i],0:mol_lens[i,0]] = False
            mask[i,0:mol_lens[i,0],0:mol_lens[i,0]] = True
            mask[i,mol_lens[i,0]:,mol_lens[i,0]:] = True
            
            mask1[i,:mol_lens[i,0],mol_lens[i,0]:2*mol_lens[i,0]] = False
            mask1[i,mol_lens[i,0]:2*mol_lens[i,0],:mol_lens[i,0]] = False
            
            mask2[i,:mol_lens[i,1],mol_lens[i,1]:2*mol_lens[i,1]] = False
            mask2[i,mol_lens[i,1]:2*mol_lens[i,1],:mol_lens[i,1]] = False
        # X1 : 50 * 42 * 42
        return X1, X2, E1, E2, A1, A2, mask1, mask2, w1, w2, mask
      
class POLYGNN_adj(nn.Module):
  def __init__(self, args, node_hidden, edge_hidden):
    super(POLYGNN_adj, self).__init__()
    self.activation = get_activation_function(args.activation)
    self.W_v = ffn(node_hidden, node_hidden,capacity=0,activation=self.activation)
    self.W_e = ffn(edge_hidden, edge_hidden,capacity=0,activation=self.activation)
    self.W_u = ffn(2*node_hidden+edge_hidden, node_hidden,capacity=0,activation=self.activation)
  
  def forward(self, X, E, A, w):
    # X [batch_size * node_size * node_hidden]
    # E [batch_size * node_size * node_size * edge_hidden]
    # A [batch_size * node_size * node_size]
    # w [batch_size * node_size]
    m = X.shape[1]
    node_map = self.W_v(X).unsqueeze(2).expand(-1,-1,m,-1) # [batch_size * target * source * node_hidden]
    edge_map = self.W_e(E)  # [batch_size * target * source * edge_hidden]
    
    out = A.unsqueeze(3) * torch.cat((node_map,edge_map),dim=3)
    out = torch.sum(out,dim=1)         # 就是在1处累加，不要再质疑了
    out = self.W_u(self.activation(torch.cat((out,X),dim=2))) * w 
    return out

'''
class GIN_adj(nn.Module):
    def __init__(self, args, node_hidden, edge_hidden):
        super(GIN_adj, self).__init__()
        self.activation0 = get_activation_function(args.activation)
        self.activation1 = get_activation_function(args.activation)
        self.args = args
        self.eps = torch.nn.Parameter(torch.zeros(1))
        self.W_0 = nn.Linear(node_hidden, node_hidden)
        self.W_1 = nn.Linear(node_hidden, node_hidden)
        self.activation = get_activation_function(args.activation)
        # below are GIN -> wDMPNN modify
        if args.gin_mode in [4,5,6,7]:                      # 残差映射
            # self.residual_mapping = nn.Linear(node_hidden,node_hidden)
            self.residual_mapping = ffn(node_hidden, node_hidden,capacity=1,activation=self.activation)
        if args.gin_mode in [1,3,5,7]:                      # 加边信息
            # self.msg_mapping = nn.Linear(node_hidden+edge_hidden,node_hidden)
            self.msg_mapping = ffn(node_hidden+edge_hidden, node_hidden,capacity=1,activation=self.activation)
        
        # below are GIN -> polyGNN modify
        if args.gin_mode in [14,15,16,17]:
            # self.edge_mapper = nn.Linear(edge_hidden,edge_hidden)
            self.edge_mapper = ffn(edge_hidden, edge_hidden,capacity=1,activation=self.activation)
        if args.gin_mode >= 10:
            if args.gin_mode in [12,13,16,17]:
                # self.residual_mapping = nn.Linear(2*node_hidden+edge_hidden,node_hidden)
                self.residual_mapping = ffn(2*node_hidden+edge_hidden, node_hidden,capacity=0,activation=self.activation)
            else:
                # self.residual_mapping = nn.Linear(node_hidden+edge_hidden,node_hidden)
                self.residual_mapping = ffn(node_hidden+edge_hidden, node_hidden,capacity=1,activation=self.activation)
        if args.gin_mode in [11,13,15,17]:
            # self.node_mapper = nn.Linear(node_hidden,node_hidden)
            self.node_mapper = ffn(node_hidden, node_hidden,capacity=1,activation=self.activation)
    
    def forward(self, X, E, A, w):
        # X [batch_size * node_size * node_hidden]
        # E [batch_size * node_size * node_size * edge_hidden]
        # A [batch_size * node_size * node_size]
        # w [batch_size * node_size]
        b = X.shape[0]
        m = X.shape[1]
        h = X.shape[2]
        
        if self.args.gin_mode < 10:                      # GIN -> wDMPNN
            if self.args.gin_mode in [1,3,5,7]:             # 加边信息
                msg = A.unsqueeze(3) * torch.cat((X.unsqueeze(2).expand(-1,-1,m,-1),E),dim=3)
            else:
                msg = A.unsqueeze(3) * X.unsqueeze(2).expand(-1,-1,m,-1)
        else:                                            # GIN -> polyGNN
            if self.args.gin_mode in [14,15,16,17]:
                E = self.edge_mapper(E)
            if self.args.gin_mode in [11,13,15,17]:
                X = self.node_mapper(X)
            msg = A.unsqueeze(3) * torch.cat((X.unsqueeze(2).expand(-1,-1,m,-1),E),dim=3)
        
        
        msg = torch.sum(msg,dim=1)
        
        
        # below are GIN -> wDMPNN modify
        if self.args.gin_mode in [1,3,5,7]:             # 加边信息
            msg = self.msg_mapping(msg)
        
        X = (1 + self.eps) * X
        
        # below are GIN -> wDMPNN modify
        if self.args.gin_mode in [4,5,6,7]:             # 残差映射
            X = self.residual_mapping(X)
            
        if self.args.gin_mode < 10:
            out = X + msg
        else:
            if self.args.gin_mode in [12,13,16,17]:
                out = self.residual_mapping(torch.cat((X,msg),dim=-1))
            else:
                out =  X + self.residual_mapping(msg)
        
        
        out = self.W_0(out)
        out = F.dropout(self.activation0(out), p=self.args.gnn_dropout, training=self.training)
        out = self.W_1(out)
        out = F.dropout(self.activation1(out), p=self.args.gnn_dropout, training=self.training)
        return out * w
'''
class GIN_adj(nn.Module):
    def __init__(self, args, node_hidden, edge_hidden):
        super(GIN_adj, self).__init__()
        self.args = args
        self.eps = torch.nn.Parameter(torch.zeros(1))
        self.W_0 = nn.Linear(node_hidden, node_hidden)
        self.W_1 = nn.Linear(node_hidden, node_hidden)
        self.activation = get_activation_function(args.activation)
    
    def forward(self, X, E, A, w):
        # X [batch_size * node_size * node_hidden]
        # E [batch_size * node_size * node_size * edge_hidden]
        # A [batch_size * node_size * node_size]
        # w [batch_size * node_size]
        b = X.shape[0]
        m = X.shape[1]
        h = X.shape[2]
            
        msg = A.unsqueeze(3) * X.unsqueeze(2).expand(-1,-1,m,-1)
        msg = torch.sum(msg,dim=1)
        
        X = (1 + self.eps) * X
        out =  X + msg
        
        out = self.W_0(out)
        out = F.dropout(self.activation(out), p=self.args.gnn_dropout, training=self.training)
        out = self.W_1(out)
        out = F.dropout(self.activation(out), p=self.args.gnn_dropout, training=self.training)
        return out * w

class poly_GIN_adj(nn.Module):
    def __init__(self, args, node_hidden, edge_hidden):
        super(poly_GIN_adj, self).__init__()
        self.args = args
        self.eps = torch.nn.Parameter(torch.zeros(1))
        self.W_0 = nn.Linear(node_hidden, node_hidden)
        self.W_1 = nn.Linear(node_hidden, node_hidden)
        self.activation = get_activation_function(args.activation)
        self.edge_mapper = ffn(edge_hidden, edge_hidden,capacity=1,activation=self.activation)
        self.residual_mapping = ffn(node_hidden+edge_hidden, node_hidden,capacity=1,activation=self.activation)
        # self.node_mapper = ffn(node_hidden, node_hidden,capacity=1,activation=self.activation)
    
    def forward(self, X, E, A):
        # X [batch_size * node_size * node_hidden]
        # E [batch_size * node_size * node_size * edge_hidden]
        # A [batch_size * node_size * node_size]
        # w [batch_size * node_size]
        b = X.shape[0]
        m = X.shape[1]
        h = X.shape[2]
            
        E = self.edge_mapper(E)
        # X = self.node_mapper(X)
        msg = A.unsqueeze(3) * torch.cat((X.unsqueeze(2).expand(-1,-1,m,-1),E),dim=3)
        msg = torch.sum(msg,dim=1)
        
        X = (1 + self.eps) * X
        out =  X + self.residual_mapping(msg)
        
        out = self.W_0(out)
        out = F.dropout(self.activation(out), p=self.args.gnn_dropout, training=self.training)
        out = self.W_1(out)
        out = F.dropout(self.activation(out), p=self.args.gnn_dropout, training=self.training)
        return out
        # return out * w

class GCN_adj(nn.Module):
    def __init__(self, args, node_hidden, edge_hidden):
        super(GCN_adj, self).__init__()
        self.activation0 = get_activation_function(args.activation)
        self.args = args
        self.W_0 = nn.Linear(node_hidden, node_hidden)
    
    def forward(self, X, E, A, w):
        # X [batch_size * node_size * node_hidden]
        # E [batch_size * node_size * node_size * edge_hidden]
        # A [batch_size * node_size * node_size]
        # w [batch_size * node_size]
        b = X.shape[0]
        m = X.shape[1]
        h = X.shape[2]
        e = torch.eye(m,device=X.device)
        A = A + e.unsqueeze(0)

        degree = torch.sum(A,dim=1)
        D = torch.pow(degree,-1/2) 
        D = torch.diag_embed(D)
        A = A * w
        output = torch.matmul(D,A)
        output = torch.matmul(output,D)
        output = torch.matmul(output,X)
        output = self.activation0(self.W_0(output))
        
        return output

class torch_adj_helper(nn.Module):
    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        super(torch_adj_helper,self).__init__()
        self.mol_graph2data_layer = mol_graph2data_adj(args)
        self.args = args
        self.encoder_type = args.encoder_type
        self.linear = nn.Linear(atom_fdim,args.hidden_size)
        self.softmax = nn.Softmax(dim=-2)
        self.alpha = nn.Parameter(torch.tensor([1.]))
        # if self.args.with_split:
        #     self.monomer_split = monomer_split(atom_fdim,bond_fdim,args.hidden_size)
        #     self.weight = nn.Parameter(torch.tensor([0.5,0.5,0.5]))
        #     self.weight.requires_grad = False       # 强制不更新
        #     self.weight_matrix = nn.Linear(3*256,256)
        if self.encoder_type == "polygnn_attn_adj":
            self.mpn = nn.ModuleList(
                [
                    POLYGNN_adj(args=args, node_hidden=args.hidden_size,edge_hidden=bond_fdim) for _ in range(args.depth)
                ]
            )
        elif self.encoder_type == "gin_attn_adj":

            self.mpn = nn.ModuleList(
                [
                    GIN_adj(args=args, node_hidden=args.hidden_size,edge_hidden=bond_fdim) for _ in range(args.depth)
                ]
            )
        elif self.encoder_type == "polygin_attn_adj":
            self.mpn = nn.ModuleList(
                [
                   
                    poly_GIN_adj(args=args, node_hidden=args.hidden_size,edge_hidden=bond_fdim) for _ in range(args.depth)
                ]
            )
            # below are GIN -> wDMPNN modify
            # if args.gin_mode in [2,3,6,7]:
            #     self.activation = get_activation_function(args.activation)
            #     self.readout_mapping = ffn(args.hidden_size*2, args.hidden_size,capacity=0,activation=self.activation)
                
        elif self.encoder_type == "gcn_attn_adj":
            self.mpn = nn.ModuleList(
                [
                    GCN_adj(args=args, node_hidden=args.hidden_size,edge_hidden=bond_fdim) for _ in range(args.depth)
                ]
            )
        self.attention_encoder = nn.ModuleList(
            [
                AttnEncoderADJ(args) for _ in range(args.depth)
            ]
        )
        # self.attention_encoder = AttnEncoderADJ(args)
        
    def forward(self,
        mol_graph: BatchMolGraph,
        atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        # X, E, A, w, mask, distances, mol_lens = self.mol_graph2data_layer(mol_graph)
        X, E, A, mask= self.mol_graph2data_layer(mol_graph)
        
        X = self.linear(X)
        A_0 = A.clone()
        X_0 = X.clone()
        if self.args.with_split:
            X1, X2, E1, E2, A1, A2, mask1, mask2, w1, w2, mask = self.monomer_split(X,E,A,w,mask,mol_lens)
            A_01 = A1.clone()
            A_02 = A2.clone()
            for i,layer in enumerate(self.mpn):
                if i == 0 or i == 1:
                    X = layer(X, E, A, w)
                    X1 = layer(X1, E1, A1, w1)
                    X2 = layer(X2, E2, A2, w2)
                else:
                    skip_x = last_last_x + X  # skip connection
                    skip_x1 = last_last_x1 + X1  # skip connection
                    skip_x2 = last_last_x2 + X2  # skip connection
                    X  = layer(skip_x, E, A, w)
                    X1 = layer(skip_x1, E1, A1, w1)
                    X2 = layer(skip_x2, E2, A2, w2)
                if i > 0:
                    last_last_x = last_x
                    last_last_x1 = last_x1
                    last_last_x2 = last_x2
                last_x = X
                last_x1 = X1
                last_x2 = X2

                # _, attn = self.attention_encoder(X,mask,distances)
                # _, attn1 = self.attention_encoder(X1,mask1,distances)
                # _, attn2 = self.attention_encoder(X2,mask2,distances)

              
                if self.args.visual and i == 0:
                    visual_matrix(A[0],self.args.checkpoint_dir,i)
                    visual_matrix(A1[0],self.args.checkpoint_dir,i,1,0,mol_lens[0,0].item())
                    visual_matrix(A2[0],self.args.checkpoint_dir,i,2,mol_lens[0,0].item(),mol_lens[0,0].item()+mol_lens[0,1].item())
                                   
                # A =   A_0 + torch.sum(attn,dim=1)
                # A1 =  A_01 + torch.sum(attn1,dim=1)
                # A2 =  A_02 + torch.sum(attn2,dim=1)
                
                if self.args.visual:
                    visual_matrix(A[0],self.args.checkpoint_dir,i+1)
                    visual_matrix(A1[0],self.args.checkpoint_dir,i+1,1,0,mol_lens[0,0].item())
                    visual_matrix(A2[0],self.args.checkpoint_dir,i+1,2,mol_lens[0,0].item(),mol_lens[0,0].item()+mol_lens[0,1].item()) 

            # X = w * X
            # X = torch.sum(X, dim=1)
            # w = torch.sum(w, dim=1)
            # X = X / w
                
            # X1 = w1 * X1
            # X1 = torch.sum(X1, dim=1)
            # w1 = torch.sum(w1, dim=1)
            # X1 = X1 / w1
            
            # X2 = w2 * X2
            # X2 = torch.sum(X2, dim=1)
            # w2 = torch.sum(w2, dim=1)
            # X2 = X2 / w2
            
            X = torch.mean(X, dim=1)
            X1 = torch.mean(X1, dim=1)
            X2 = torch.mean(X2, dim=1)
            
            X = self.weight[0] * X + self.weight[1] * X1 + self.weight[2] * X2
            # X = self.weight_matrix(torch.cat((X  , X1 , X2 ),dim=-1))   
        else:
            #if self.args.visual == -1:
                #return A
            for i,layer in enumerate(self.mpn):
                last_x = X.clone()
                # X = layer(X, E, A, w)
                # X, attn = self.attention_encoder[i](X+last_x,mask,distances,A)
                # X += last_x
                # if self.args.visual and i == 0:
                #     visual_matrix(A[0],self.args.checkpoint_dir,i)
                X = layer(X, E, A)
                #if self.args.visual == i+11:
                    #return X
                X = X + last_x
                _, attn = self.attention_encoder[i](X,mask,A)
                A =  A_0 + self.alpha * torch.sum(attn,dim=1)
                # if self.args.visual:
                #     visual_matrix(A[0],self.args.checkpoint_dir,i+1)
                #if self.args.visual == i + 1:
                    #return A
            # below are GIN -> wDMPNN modify
            # if self.encoder_type == "gin_attn_adj" and self.args.gin_mode in [2,3,6,7]:
            #     X = self.readout_mapping(torch.cat((X_0,X),dim=-1))
                
            A = torch.mean(A,dim=1)
            X = X * A.unsqueeze(-1)
            X = torch.sum(X,dim=1) / torch.sum(A,dim=1).unsqueeze(-1)

        return X
       
class GAIN(nn.Module):
    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        super(GAIN, self).__init__()
        self.gin = GIN(in_channels=atom_fdim,hidden_channels=args.hidden_size,num_layers=args.depth,dropout=args.gnn_dropout,act=args.activation)
        self.bn1 = torch.nn.BatchNorm1d(args.hidden_size)
        self.gat = GAT(in_channels=args.hidden_size,hidden_channels=args.hidden_size,num_layers=args.depth,dropout=args.drognn_dropoutpout,act=args.activation)

    def forward(self,x,edge_index,edge_attr,edge_weight,batch):
        x = self.gin(x=x, edge_index=edge_index, edge_attr=edge_attr,edge_weight=edge_weight, batch=batch)
        x = F.relu(x)
        x = self.bn1(x)
        return F.relu(self.gat(x=x, edge_index=edge_index, edge_attr=edge_attr,edge_weight=edge_weight, batch=batch))

class pyG_helper(nn.Module):
    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        super(pyG_helper, self).__init__()
        self.mol_graph2data_layer = mol_graph2data(args)
        self.args = args
        self.encoder_type = args.encoder_type
        if args.encoder_type in ["polygin", "polygin_attn"]:
            self.mpn = POLYGIN(in_channels=atom_fdim,hidden_channels=args.hidden_size, \
                                num_layers=args.depth,dropout=args.gnn_dropout,activation=get_activation_function(args.activation), \
                                edge_hidden=bond_fdim,node_hidden=args.hidden_size)
    def forward(self,
                mol_graph: BatchMolGraph,
                atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        f_atoms[1:,:].to(self.args.device),\
               edge_index, \
               f_bonds[1:,:].to(self.args.device),\
               b2revb_individual, \
               batch, ptr
        """
        x, edge_index, edge_attr, b2revb_individual, batch, ptr = self.mol_graph2data_layer(mol_graph) 
        z_table = AtomicNumberTable([1, 6, 7, 8, 9]) 

        if self.args.encoder_type in ["polygin", "polygin_attn", "polygin_pe"]:
            poly_vec = self.mpn(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        else:
            poly_vec = self.mpn(x=x, edge_index=edge_index, edge_attr=edge_attr,batch=batch)

        poly_vec = scatter_sum(poly_vec, batch, dim=0)
      
        return poly_vec

class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim(overwrite_default_atom=args.overwrite_default_atom_features)
        
        self.bond_fdim = bond_fdim or get_bond_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                    overwrite_default_bond=args.overwrite_default_bond_features,
                                                    atom_messages=args.atom_messages)

        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.atom_descriptors = args.atom_descriptors
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        self.overwrite_default_bond_features = args.overwrite_default_bond_features

        if self.features_only:
            return
        if args.encoder_type in ["wDMPNN","wDMPNN_origin_attn","wDMPNN_attn","wDMPNN_pe"]:
            encoder = MPNEncoder
        elif "adj" in args.encoder_type:
            encoder = torch_adj_helper
        else:
            encoder = pyG_helper  

        if args.mpn_shared: # 只在number_of_molecules>1时有效
            self.encoder = nn.ModuleList([encoder(args, self.atom_fdim, self.bond_fdim)] * args.number_of_molecules)
        else:
            self.encoder = nn.ModuleList([encoder(args, self.atom_fdim, self.bond_fdim)
                                          for _ in range(args.number_of_molecules)])

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None,
                # start_time=None,
                # logger=None
                ) -> torch.FloatTensor:

        """
        Encodes a batch of molecules.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        # debug = logger.debug if logger is not None else print
        if type(batch[0]) != BatchMolGraph:
            # Group first molecules, second molecules, etc for mol2graph
            batch = [[mols[i] for mols in batch] for i in range(len(batch[0]))]

            # TODO: handle atom_descriptors_batch with multiple molecules per input
            if self.atom_descriptors == 'feature':
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        atom_features_batch=atom_features_batch,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            elif bond_features_batch is not None:
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')

                batch = [
                    mol2graph(
                        mols=b,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            else:
                batch = [mol2graph(b) for b in batch]
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)

            if self.features_only:
                return features_batch

        if self.atom_descriptors == 'descriptor':
            if len(batch) > 1:
                raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                          'per input (i.e., number_of_molecules = 1).')

            encodings = [enc(ba, atom_descriptors_batch) for enc, ba in zip(self.encoder, batch)]
        else:
            encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]
        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)
            output = torch.cat([output, features_batch], dim=1)
        return output
