import copy
from abc import ABC, abstractmethod

import numpy as np
import torch

from util import functional as smolF
from models.lr_mechanisms import (
    CoordLatentLinResTransfer,
    CoordLatentSetMixTransfer,
    CoordNodeLatentNFFTransfer,
    EdgeLatentBaseTransfer,
    EdgeLatentGatedTransfer,
    EdgeLatentLinResTransfer,
    EdgeLatentTransfer,
    NFFUpdateTransfer,
    NodeLatentBaseTransfer,
    NodeLatentIdentityTransfer,
    NodeLatentLinResTransfer,
    VLinResTransfer,
    VSetMixTransfer,
)


def adj_to_attn_mask(adj_matrix, pos_inf=False):
    """Assumes adj_matrix is only 0s and 1s"""

    inf = float("inf") if pos_inf else float("-inf")
    attn_mask = torch.zeros_like(adj_matrix.float())
    attn_mask[adj_matrix == 0] = inf

    # Ensure nodes with no connections (fake nodes) don't have all -inf in the attn mask
    # Otherwise we would have problems when softmaxing
    n_nodes = adj_matrix.sum(dim=-1)
    attn_mask[n_nodes == 0] = 0.0

    return attn_mask


# *************************************************************************************************
# *********************************** Helper Classes **********************************************
# *************************************************************************************************



class CoordNorm(torch.nn.Module):
    """Coordinate normalisation layer for coordinate sets with inductive bias towards molecules

    This layer allows 4 different types of coordinate normalisation (defined in the norm argument):
        1. 'none' - The coordinates are zero-centred and multiplied by learnable weights
        2. 'gvp' - Coords are zero-centred, scaled by learnable weights and each is scaled by sqrt(n_sets) / ||x_i||_2
        3. 'length' - Coords are zero-centred, multiplied by learnable weights and scaled by 1 / avg vector length

    Note that 'length' provides the same coordinate normalisation that is commonly used in current models but adapted
    to multiple coordinate sets, thereby allowing easier comparison to existing approaches.
    """

    def __init__(self, n_coord_sets, norm="length", eps=1e-6):
        super().__init__()

        norm = "none" if norm is None else norm
        if norm not in ["none", "gvp", "length"]:
            raise ValueError(f"Unknown normalisation type '{norm}'")

        self.n_coord_sets = n_coord_sets
        self.norm = norm
        self.eps = eps

        self.set_weights = torch.nn.Parameter(torch.ones((1, n_coord_sets, 1, 1)))

    def forward(self, coord_sets, node_mask):
        """Apply coordinate normlisation layer

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise

        Returns:
            torch.Tensor: Normalised coords, shape [batch_size, n_sets, n_nodes, 3]
        """

        # Zero the CoM in case it isn't already
        coord_sets = smolF.zero_com(coord_sets, node_mask)
        coord_sets = coord_sets * node_mask.unsqueeze(-1)

        n_atoms = node_mask.sum(dim=-1, keepdim=True)
        lengths = torch.linalg.vector_norm(coord_sets, dim=-1)

        if self.norm == "length":
            scaled_lengths = lengths.sum(dim=2, keepdim=True) / n_atoms
            coord_div = scaled_lengths.unsqueeze(-1) + self.eps

        elif self.norm == "gvp":
            coord_div = (lengths.unsqueeze(-1) + self.eps) / np.sqrt(self.n_coord_sets)

        else:
            coord_div = torch.ones_like(coord_sets)

        coord_sets = (coord_sets * self.set_weights) / coord_div
        coord_sets = coord_sets * node_mask.unsqueeze(-1)
        return coord_sets

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


class EdgeMessages(torch.nn.Module):
    def __init__(self, d_model, d_message, d_out, n_coord_sets, d_ff=None, d_edge=None, eps=1e-6):
        super().__init__()

        edge_feats = 0 if d_edge is None else d_edge
        d_ff = d_out if d_ff is None else d_ff

        extra_feats = n_coord_sets + edge_feats
        in_feats = (d_message * 2) + extra_feats

        self.n_coord_sets = n_coord_sets
        self.d_edge = d_edge
        self.eps = eps

        self.coord_norm = CoordNorm(n_coord_sets, norm="none")
        self.node_norm = torch.nn.LayerNorm(d_model)
        self.edge_norm = torch.nn.LayerNorm(d_edge) if d_edge is not None else None

        self.node_proj = torch.nn.Linear(d_model, d_message)
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_ff), torch.nn.SiLU(inplace=False), torch.nn.Linear(d_ff, d_out)
        )

    def forward(self, coords, node_feats, node_mask, edge_feats=None):
        """Compute edge messages

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node features, shape [batch_size, n_nodes, d_model]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise
            edge_feats (torch.Tensor): Incoming edge features, shape [batch_size, n_nodes, n_nodes, d_edge]

        Returns:
            torch.Tensor: Edge messages tensor, shape [batch_size, n_nodes, n_nodes, d_out]
        """

        batch_size, n_nodes, _ = tuple(node_feats.shape)

        if edge_feats is not None and self.d_edge is None:
            raise ValueError("edge_feats was provided but the model was initialised with d_edge as None.")

        if edge_feats is None and self.d_edge is not None:
            raise ValueError("The model was initialised with d_edge but no edge feats were provided to forward fn.")

        node_feats = self.node_norm(node_feats)

        coords = self.coord_norm(coords, node_mask).flatten(0, 1)
        coord_dotprods = torch.bmm(coords, coords.transpose(1, 2))
        coord_feats = coord_dotprods.unflatten(0, (-1, self.n_coord_sets)).movedim(1, -1)

        # Project to smaller dimension and create pairwise node features
        node_feats = self.node_proj(node_feats)
        node_feats_start = node_feats.unsqueeze(2).expand(batch_size, n_nodes, n_nodes, -1)
        node_feats_end = node_feats.unsqueeze(1).expand(batch_size, n_nodes, n_nodes, -1)
        node_pairs = torch.cat((node_feats_start, node_feats_end), dim=-1)

        in_edge_feats = torch.cat((node_pairs, coord_feats), dim=3)
        if edge_feats is not None:
            edge_feats = self.edge_norm(edge_feats)
            in_edge_feats = torch.cat((in_edge_feats, edge_feats), dim=-1)

        return self.message_mlp(in_edge_feats)


class NodeAttention(torch.nn.Module):
    def __init__(self, d_model, n_attn_heads, d_attn=None):
        super().__init__()

        d_attn = d_model if d_attn is None else d_attn
        d_head = d_model // n_attn_heads

        if d_attn % n_attn_heads != 0:
            raise ValueError("n_attn_heads must divide d_model (or d_attn if provided) exactly.")

        self.d_model = d_model
        self.d_attn = d_attn
        self.n_attn_heads = n_attn_heads
        self.d_head = d_head

        self.feat_norm = torch.nn.LayerNorm(d_model)
        self.in_proj = torch.nn.Linear(d_model, d_attn)
        self.out_proj = torch.nn.Linear(d_attn, d_model)

    def forward(self, node_feats, messages, adj_matrix):
        """Accumulate edge messages to each node using attention-based message passing

        Args:
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            messages (torch.Tensor): Messages tensor, shape [batch_size, n_nodes, n_nodes, d_message]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_nodes, n_nodes]

        Returns:
            torch.Tensor: Accumulated node features, shape [batch_size, n_nodes, d_model]
        """

        attn_mask = adj_to_attn_mask(adj_matrix)
        messages = messages + attn_mask.unsqueeze(3)
        attentions = torch.softmax(messages, dim=2)

        node_feats = self.feat_norm(node_feats)
        proj_feats = self.in_proj(node_feats)
        head_feats = proj_feats.unflatten(-1, (self.n_attn_heads, self.d_head))

        # Put n_heads into the batch dim for both the features and the attentions
        # head_feats shape [B * n_heads, n_nodes, d_head]
        # attentions shape [B * n_heads, n_nodes, n_nodes]
        head_feats = head_feats.movedim(-2, 1).flatten(0, 1)
        attentions = attentions.movedim(-1, 1).flatten(0, 1)

        attn_out = torch.bmm(attentions, head_feats)

        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        weights = torch.sqrt((attentions**2).sum(dim=-1))
        attn_out = attn_out * weights.unsqueeze(-1)

        attn_out = attn_out.unflatten(0, (-1, self.n_attn_heads))
        attn_out = attn_out.movedim(1, -2).flatten(2, 3)
        return self.out_proj(attn_out)


class CoordAttention(torch.nn.Module):
    def __init__(self, n_coord_sets, proj_sets=None, coord_norm="length", eps=1e-6):
        super().__init__()

        proj_sets = n_coord_sets if proj_sets is None else proj_sets

        self.eps = eps

        self.coord_norm = CoordNorm(n_coord_sets, norm=coord_norm)
        self.coord_proj = torch.nn.Linear(n_coord_sets, proj_sets, bias=False)
        self.attn_proj = torch.nn.Linear(proj_sets, n_coord_sets, bias=False)

    def forward(self, coord_sets, messages, adj_matrix, node_mask):
        """Compute an attention update for coordinate sets

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            messages (torch.Tensor): Messages tensor, shape [batch_size, n_nodes, n_nodes, proj_sets]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_nodes, n_nodes]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise

        Returns:
            torch.Tensor: Updated coordinate sets, shape [batch_size, n_sets, n_nodes, 3]
        """

        coord_sets = self.coord_norm(coord_sets, node_mask)
        proj_coord_sets = self.coord_proj(coord_sets.transpose(1, -1))

        # proj_coord_sets shape [B, 3, N, P]
        # norm_dists shape [B, 1, N, N, P]
        vec_dists = proj_coord_sets.unsqueeze(3) - proj_coord_sets.unsqueeze(2)
        lengths = torch.linalg.vector_norm(vec_dists, dim=1, keepdim=True)
        norm_dists = vec_dists / (lengths + self.eps)

        attn_mask = adj_to_attn_mask(adj_matrix)
        messages = messages + attn_mask.unsqueeze(3)
        attentions = torch.softmax(messages, dim=2)

        # Dim 1 is currently 1 on dists so we need to unsqueeze attentions
        updates = norm_dists * attentions.unsqueeze(1)
        updates = updates.sum(dim=3)

        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        weights = torch.sqrt((attentions**2).sum(dim=2))
        updates = updates * weights.unsqueeze(1)

        # updates shape [B, 3, N, P] -> [B, S, N, 3]
        return self.attn_proj(updates).transpose(1, -1)


class LengthsMLP(torch.nn.Module):
    def __init__(self, d_model, n_coord_sets, d_ff=None):
        super().__init__()

        d_ff = d_model * 4 if d_ff is None else d_ff

        self.node_ff = torch.nn.Sequential(
            torch.nn.Linear(d_model + n_coord_sets, d_ff), torch.nn.SiLU(inplace=False), torch.nn.Linear(d_ff, d_model)
        )

    def forward(self, coord_sets, node_feats):
        """Pass data through the layer

        Assumes coords and node_feats have already been normalised

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]

        Returns:
            torch.Tensor: Updated node features, shape [batch_size, n_nodes, d_model]
        """

        lengths = torch.linalg.vector_norm(coord_sets, dim=-1).movedim(1, -1)
        in_feats = torch.cat((node_feats, lengths), dim=2)
        return self.node_ff(in_feats)


class EquivariantMLP(torch.nn.Module):
    def __init__(self, d_model, n_coord_sets, proj_sets=None):
        super().__init__()

        proj_sets = n_coord_sets if proj_sets is None else proj_sets

        self.node_proj = torch.nn.Sequential(
            torch.nn.Linear(d_model, proj_sets), torch.nn.SiLU(inplace=False), torch.nn.Linear(proj_sets, proj_sets)
        )
        self.coord_proj = torch.nn.Linear(n_coord_sets, proj_sets, bias=False)
        self.attn_proj = torch.nn.Linear(proj_sets, n_coord_sets, bias=False)

    def forward(self, coord_sets, node_feats):
        """Pass data through the layer

        Assumes coords and node_feats have already been normalised

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]

        Returns:
            torch.Tensor: Updated coord_sets, shape [batch_size, n_sets, n_nodes, 3]
        """

        # inv_feats shape [B, 1, N, P]
        # proj_sets shape [B, 3, N, P]
        inv_feats = self.node_proj(node_feats).unsqueeze(1)
        proj_sets = self.coord_proj(coord_sets.transpose(1, -1))

        # Outer product with invariant features is equivariant, then sum over original coord sets
        attentions = inv_feats.unsqueeze(-1) * proj_sets.unsqueeze(-2)
        attentions = attentions.sum(-1)

        return self.attn_proj(attentions).transpose(1, -1)


class NodeFeedForward(torch.nn.Module):
    def __init__(self, d_model, n_coord_sets, d_ff=None, proj_sets=None, coord_norm="length"):
        super().__init__()

        self.node_norm = torch.nn.LayerNorm(d_model)
        self.coord_norm = CoordNorm(n_coord_sets, norm=coord_norm)

        self.invariant_mlp = LengthsMLP(d_model, n_coord_sets, d_ff=d_ff)
        self.equivariant_mlp = EquivariantMLP(d_model, n_coord_sets, proj_sets=proj_sets)

    def forward(self, coord_sets, node_feats, node_mask):
        """Pass data through the layer

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise

        Returns:
            torch.Tensor, torch.Tensor: Updates to coords and node features
        """

        node_feats = self.node_norm(node_feats)
        coord_sets = self.coord_norm(coord_sets, node_mask)

        out_node_feats = self.invariant_mlp(coord_sets, node_feats)
        out_coord_sets = self.equivariant_mlp(coord_sets, node_feats)

        return out_coord_sets, out_node_feats


class BondRefine(torch.nn.Module):
    def __init__(self, d_model, d_message, d_edge, d_ff=None):
        super().__init__()

        d_ff = d_message if d_ff is None else d_ff
        in_feats = (2 * d_message) + d_edge + 2

        self.coord_norm = CoordNorm(1, norm="none")
        self.node_norm = torch.nn.LayerNorm(d_model)
        self.edge_norm = torch.nn.LayerNorm(d_edge)

        self.node_proj = torch.nn.Linear(d_model, d_message)
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_ff), torch.nn.SiLU(inplace=False), torch.nn.Linear(d_ff, d_edge)
        )

    def forward(self, coords, node_feats, node_mask, edge_feats):
        """Refine the bond predictions with a message passing layer that only updates bonds

        Args:
            coords (torch.Tensor): Coordinate tensor without coord sets, shape [batch_size, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_nodes], 1 for real, 0 otherwise
            edge_feats (torch.Tensor): Current edge features, shape [batch_size, n_nodes, n_nodes, d_edge]

        Returns:
            torch.Tensor: Bond predictions tensor, shape [batch_size, n_nodes, n_nodes, n_bond_types]
        """

        assert len(coords.shape) == 3

        batch_size, n_nodes, _ = tuple(node_feats.shape)

        # Calculate distances and dot products
        coords = self.coord_norm(coords.unsqueeze(1), node_mask.unsqueeze(1)).squeeze(1)
        coord_diffs = coords.unsqueeze(2) - coords.unsqueeze(1)
        dists = (coord_diffs * coord_diffs).sum(dim=-1).unsqueeze(-1)
        coord_dotprods = torch.bmm(coords, coords.transpose(1, 2)).unsqueeze(-1)

        # Project to smaller dimension and create pairwise node features
        node_feats = self.node_proj(self.node_norm(node_feats))
        node_feats_i = node_feats.unsqueeze(2).expand(batch_size, n_nodes, n_nodes, -1)
        node_feats_j = node_feats.unsqueeze(1).expand(batch_size, n_nodes, n_nodes, -1)
        node_pairs = torch.cat((node_feats_i, node_feats_j), dim=-1)

        edge_feats = self.edge_norm(edge_feats)
        in_feats = torch.cat((node_pairs, dists, coord_dotprods, edge_feats), dim=3)
        return self.message_mlp(in_feats)


# *************************************************************************************************
# ********************************** Equivariant Layers *******************************************
# *************************************************************************************************
class EquiMessagePassingLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_message,
        n_coord_sets,
        n_attn_heads=None,
        d_message_hidden=None,
        d_edge_in=None,
        d_edge_out=None,
        coord_norm="length",
        eps=1e-6,
    ):
        super().__init__()

        n_attn_heads = d_message if n_attn_heads is None else n_attn_heads
        if d_model != ((d_model // n_attn_heads) * n_attn_heads):
            raise ValueError(f"n_attn_heads must exactly divide d_model, got {n_attn_heads} and {d_model}")

        self.d_model = d_model
        self.d_message = d_message
        self.n_coord_sets = n_coord_sets
        self.n_attn_heads = n_attn_heads
        self.d_message_hidden = d_message_hidden
        self.d_edge_in = d_edge_in
        self.d_edge_out = d_edge_out
        self.d_coord_message = n_coord_sets
        self.eps = eps

        d_ff = d_model * 4
        d_attn = d_model
        d_message_out = n_attn_heads + self.d_coord_message
        d_message_out = d_message_out + d_edge_out if d_edge_out is not None else d_message_out

        if d_edge_in is not None:
            self.edge_feat_norm = torch.nn.LayerNorm(d_edge_in)

        self.node_ff = NodeFeedForward(
            d_model,
            n_coord_sets,
            d_ff=d_ff,
            proj_sets=d_message,
            coord_norm=coord_norm,
        )
        self.message_ff = EdgeMessages(
            d_model, d_message, d_message_out, n_coord_sets, d_ff=d_message_hidden, d_edge=d_edge_in, eps=eps
        )
        self.coord_attn = CoordAttention(n_coord_sets, self.d_coord_message, coord_norm=coord_norm, eps=eps)
        self.node_attn = NodeAttention(d_model, n_attn_heads, d_attn=d_attn)

    @property
    def hparams(self):
        return {
            "d_model": self.d_model,
            "d_message": self.d_message,
            "n_coord_sets": self.n_coord_sets,
            "n_attn_heads": self.n_attn_heads,
            "d_message_hidden": self.d_message_hidden,
        }

    def forward(self, coords, node_feats, adj_matrix, node_mask, edge_feats=None):
        """Pass data through the layer

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node features, shape [batch_size, n_nodes, d_model]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_nodes, n_nodes]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise
            edge_feats (torch.Tensor): Incoming edge features, shape [batch_size, n_nodes, n_nodes, d_edge_in]

        Returns:
            Either a two-tuple of the new node coordinates and the new node features, or a three-tuple of the new
            node coords, new node features and new edge features.
        """

        if edge_feats is not None and self.d_edge_in is None:
            raise ValueError("edge_feats was provided but the model was initialised with d_edge_in as None.")

        if edge_feats is None and self.d_edge_in is not None:
            raise ValueError("The model was initialised with d_edge_in but no edge feats were provided to forward.")

        coord_updates, node_updates = self.node_ff(coords, node_feats, node_mask)
        coords = coords + coord_updates
        node_feats = node_feats + node_updates

        messages = self.message_ff(coords, node_feats, node_mask, edge_feats=edge_feats)
        node_messages = messages[:, :, :, : self.n_attn_heads]
        coord_messages = messages[:, :, :, self.n_attn_heads : (self.n_attn_heads + self.d_coord_message)]

        node_feats = node_feats + self.node_attn(node_feats, node_messages, adj_matrix)
        coords = coords + self.coord_attn(coords, coord_messages, adj_matrix, node_mask)

        if self.d_edge_out is not None:
            edge_out = messages[:, :, :, (self.n_attn_heads + self.d_coord_message) :]
            edge_out = edge_feats + edge_out if edge_feats is not None else edge_out
            return coords, node_feats, edge_out

        return coords, node_feats


# *************************************************************************************************
# ************************************* Dynamics Models *******************************************
# *************************************************************************************************


class EquiInvDynamics(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_message,
        n_coord_sets,
        n_layers,
        n_attn_heads=None,
        d_message_hidden=None,
        d_edge=None,
        bond_refine=True,
        self_cond=False,
        coord_norm="length",
        eps=1e-6,
    ):
        super().__init__()

        extra_layers = 2 if d_edge is not None else 0
        if extra_layers > n_layers:
            raise ValueError("n_layers is too small.")

        n_attn_heads = d_message if n_attn_heads is None else n_attn_heads
        if d_model != ((d_model // n_attn_heads) * n_attn_heads):
            raise ValueError(f"n_attn_heads must exactly divide d_model, got {n_attn_heads} and {d_model}")

        self._hparams = {
            "d_model": d_model,
            "d_message": d_message,
            "n_coord_sets": n_coord_sets,
            "n_layers": n_layers,
            "n_attn_heads": n_attn_heads,
            "d_message_hidden": d_message_hidden,
            "d_edge": d_edge,
            "bond_refine": bond_refine,
            "self_cond": self_cond,
            "coord_norm": coord_norm,
            "eps": eps,
        }

        self.d_model = d_model
        self.n_coord_sets = n_coord_sets
        self.d_edge = d_edge
        self.bond_refine = bond_refine and d_edge is not None
        self.self_cond = self_cond

        core_layer = EquiMessagePassingLayer(
            d_model,
            d_message,
            n_coord_sets,
            n_attn_heads=n_attn_heads,
            d_message_hidden=d_message_hidden,
            coord_norm=coord_norm,
            eps=eps,
        )
        layers = self._get_clones(core_layer, n_layers - extra_layers)

        if d_edge is not None:
            # Pass d_message_hidden as None so that these layers will have the same feats as their output
            in_layer = EquiMessagePassingLayer(
                d_model,
                d_message,
                n_coord_sets,
                n_attn_heads=n_attn_heads,
                d_message_hidden=None,
                d_edge_in=d_edge,
                coord_norm=coord_norm,
                eps=eps,
            )
            out_layer = EquiMessagePassingLayer(
                d_model,
                d_message,
                n_coord_sets,
                n_attn_heads=n_attn_heads,
                d_message_hidden=None,
                d_edge_out=d_edge,
                coord_norm=coord_norm,
                eps=eps,
            )
            layers = [in_layer] + layers + [out_layer]

        self.layers = torch.nn.ModuleList(layers)

        self.final_ff_block = NodeFeedForward(d_model, n_coord_sets, coord_norm=coord_norm)
        self.coord_norm = CoordNorm(n_coord_sets, norm=coord_norm)
        self.feat_norm = torch.nn.LayerNorm(d_model)

        in_coord_sets = 2 if self_cond else 1
        self.coord_proj = torch.nn.Linear(in_coord_sets, n_coord_sets, bias=False)
        self.coord_head = torch.nn.Linear(n_coord_sets, 1, bias=False)

        if d_edge is not None:
            self.bond_norm = torch.nn.LayerNorm(d_edge)

        if self.bond_refine:
            self.refine_layer = BondRefine(d_model, d_message, d_edge)

    @property
    def hparams(self):
        return self._hparams

    def forward(
        self,
        coords,
        inv_feats,
        adj_matrix,
        atom_mask=None,
        edge_feats=None,
        cond_coords=None,
        prev_edge_latent=None,
        latrec_e_readout=None,
        pair_mask=None,
        latrec_v=None,
        prev_coord_latent=None,
        prev_node_latent=None,
    ):
        """Generate molecular coordinates and atom features

        Args:
            coords (torch.Tensor): Input coordinates, shape [batch_size, n_atoms, 3]
            inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, d_model]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_atoms, n_atoms], 1 for connected
            atom_mask (torch.Tensor, Optional): Mask for fake atoms, shape [batch_size, n_atoms], 1 for real atoms
            edge_feats (torch.Tensor, Optional): In edge features, shape [batch_size, n_nodes, n_nodes, d_edge]
            cond_coords (torch.Tensor, Optional): Conditional coords, shape [batch_size, n_nodes, 3]
            prev_edge_latent (torch.Tensor, Optional): Carried edge representation from previous step
            latrec_e_readout (EdgeLatentTransfer, Optional): Readout transfer module before BondRefine (E2)
            pair_mask (torch.Tensor, Optional): Real atom pair mask, shape [batch_size, n_atoms, n_atoms]
            latrec_v (CoordLatentTransfer, Optional): Coordinate transfer module before final_ff_block
            prev_coord_latent (torch.Tensor, Optional): Carried coordinate set representation from previous step
            prev_node_latent (torch.Tensor, Optional): Carried node scalar representation from previous step

        Returns:
            (coords, atom feats, [edge feats], v_latent)
        """

        if edge_feats is not None and self.d_edge is None:
            raise ValueError("edge_feats was provided but the model was initialised with d_edge as None.")

        if edge_feats is None and self.d_edge is not None:
            raise ValueError("The model was initialised with d_edge but no edge feats were provided to forward.")

        if cond_coords is not None and not self.self_cond:
            raise ValueError("cond_coords was provided but the model was initialised with self_cond as False.")

        if cond_coords is None and self.self_cond:
            raise ValueError("The model was initialsed with self_cond but cond_coords was not provided.")

        # Project single coord set into a multiple learnable coord sets, while maintaining equivariance
        coords = torch.stack((coords, cond_coords)) if cond_coords is not None else coords.unsqueeze(0)
        coords = self.coord_proj(coords.movedim(0, -1)).movedim(-1, 1)

        atom_mask = atom_mask.unsqueeze(1).expand(-1, self.n_coord_sets, -1)
        coords = coords * atom_mask.unsqueeze(-1)

        # Update coords and node feats using the model layers
        for layer in self.layers:
            out = layer(coords, inv_feats, adj_matrix, atom_mask, edge_feats=edge_feats)
            if len(out) == 2:
                coords, inv_feats = out
                edge_feats = None

            elif len(out) == 3:
                coords, inv_feats, edge_feats = out

        # Coordinate latent recursion: tap coords before final_ff_block (full 64 sets rank)
        transfer_v = latrec_v or getattr(self, "latrec_v", None)
        if transfer_v is not None and prev_coord_latent is not None:
            coords, inv_feats = transfer_v(
                coords, prev_coord_latent, atom_mask, inv_feats=inv_feats, prev_inv_feats=prev_node_latent
            )
        v_latent = coords * atom_mask.unsqueeze(-1)

        # Apply a final feedforward block and project coord sets to single coord set
        coords, inv_feats = self.final_ff_block(coords, inv_feats, atom_mask)
        out_coords = self.coord_norm(coords, atom_mask)
        out_coords = self.coord_head(out_coords.transpose(1, -1))
        out_coords = out_coords.transpose(1, -1).squeeze(1)

        # Direct readout injection before BondRefine (E2 variant)
        transfer_module = latrec_e_readout or getattr(self, "latrec_e_readout", None)
        if transfer_module is not None and prev_edge_latent is not None:
            if pair_mask is None:
                m = atom_mask[:, 0, :] if atom_mask.dim() == 3 else atom_mask
                pair_mask = (m.unsqueeze(2) * m.unsqueeze(1)).float()
            edge_feats = transfer_module(edge_feats, prev_edge_latent, pair_mask)

        if self.bond_refine:
            atom_mask = atom_mask[:, 0, :]
            edge_feats = self.refine_layer(out_coords, inv_feats, atom_mask, edge_feats)

        inv_feats = self.feat_norm(inv_feats)

        if self.d_edge is None:
            return out_coords, inv_feats, v_latent

        edge_feats = self.bond_norm(edge_feats)
        return out_coords, inv_feats, edge_feats, v_latent

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]


# *********************************************************************************************************************
# ****************************************** Molecular Generation Models **********************************************
# *********************************************************************************************************************


class MolecularGenerator(ABC, torch.nn.Module):
    """Interface for molecular generation classes"""

    def __init__(self, **kwargs):
        super().__init__()
        self._hparams = kwargs

    @property
    def hparams(self):
        return self._hparams

    @abstractmethod
    def forward(
        self,
        coords,
        inv_feats,
        edge_feats=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
        atom_mask=None,
    ):
        pass


class SemlaGenerator(MolecularGenerator):
    def __init__(
        self,
        d_model,
        dynamics,
        vocab_size,
        n_atom_feats,
        d_edge=None,
        n_edge_types=None,
        self_cond=False,
        size_emb=64,
        max_atoms=256,
        latent_recursion=False,
        latrec_transfer="identity",
    ):

        hparams = {
            "d_model": d_model,
            "vocab_size": vocab_size,
            "n_atom_feats": n_atom_feats,
            "d_edge": d_edge,
            "n_edge_types": n_edge_types,
            "self_cond": self_cond,
            "size_emb": size_emb,
            "max_atoms": max_atoms,
            "latent_recursion": latent_recursion,
            "latrec_transfer": latrec_transfer,
            **dynamics.hparams,
        }

        super().__init__(**hparams)

        self.self_cond = self_cond
        self.latent_recursion = latent_recursion
        self.latrec_transfer = latrec_transfer
        self.d_edge = d_edge

        if self.latent_recursion:
            self.latrec_v = None
            if latrec_transfer in ["identity", "node_lr_base", "node_lr_identity"]:
                self.latrec_s = NodeLatentIdentityTransfer()
                self.latrec_mlp = None
                self.latrec_e = None
                self.latrec_e_readout = None
            elif latrec_transfer in ["node_lr_lin_res"]:
                self.latrec_s = NodeLatentLinResTransfer(d_model)
                self.latrec_mlp = None
                self.latrec_e = None
                self.latrec_e_readout = None
            elif any(v in latrec_transfer for v in ["v_setmix", "coord_setmix"]):
                n_coord_sets = getattr(dynamics, "n_coord_sets", 64)
                coord_norm = dynamics.hparams.get("coord_norm", "length") if hasattr(dynamics, "hparams") else "length"
                eps = dynamics.hparams.get("eps", 1e-6) if hasattr(dynamics, "hparams") else 1e-6
                self.latrec_v = CoordLatentSetMixTransfer(n_coord_sets=n_coord_sets, coord_norm=coord_norm, eps=eps)
                self.latrec_s = NodeLatentIdentityTransfer() if "+identity" in latrec_transfer else None
                self.latrec_mlp = None
                self.latrec_e = None
                self.latrec_e_readout = None
            elif any(v in latrec_transfer for v in ["v_lin_res", "coord_lin_res"]):
                n_coord_sets = getattr(dynamics, "n_coord_sets", 64)
                self.latrec_v = CoordLatentLinResTransfer(n_coord_sets=n_coord_sets)
                self.latrec_s = NodeLatentIdentityTransfer() if "+identity" in latrec_transfer else None
                self.latrec_mlp = None
                self.latrec_e = None
                self.latrec_e_readout = None
            elif any(v in latrec_transfer for v in ["nff_update", "coord_nff"]):
                n_coord_sets = getattr(dynamics, "n_coord_sets", 64)
                coord_norm = dynamics.hparams.get("coord_norm", "length") if hasattr(dynamics, "hparams") else "length"
                self.latrec_v = CoordNodeLatentNFFTransfer(d_model=d_model, n_coord_sets=n_coord_sets, coord_norm=coord_norm)
                self.latrec_s = NodeLatentIdentityTransfer() if "+identity" in latrec_transfer else None
                self.latrec_mlp = None
                self.latrec_e = None
                self.latrec_e_readout = None
            elif latrec_transfer in [
                "node_edge_e1_lin_res",
                "node_lr_edge_lr_e1_lin_res",
                "node_lin_res_edge_e1_lin_res",
                "edge_lr_e1_lin_res+node_lr_lin_res",
                "node_lr_lin_res+edge_lr_e1_lin_res",
                "edge_lr_e1_lin_res+node_lin_res",
            ]:
                self.latrec_s = NodeLatentLinResTransfer(d_model)
                self.latrec_mlp = None
                self.latrec_e = EdgeLatentLinResTransfer(d_edge) if d_edge is not None else None
                self.latrec_e_readout = None
            elif "edge_lr" in latrec_transfer or latrec_transfer == "both":
                self.latrec_s = NodeLatentIdentityTransfer() if (latrec_transfer == "both" or "+identity" in latrec_transfer) else None
                self.latrec_mlp = None
                if "lin_res" in latrec_transfer:
                    transfer_fn = lambda: EdgeLatentLinResTransfer(d_edge) if d_edge is not None else None
                elif "gated" in latrec_transfer or latrec_transfer in ["edge_lr", "edge_lr_e1", "edge_lr_e2", "both", "edge_lr+identity"]:
                    transfer_fn = lambda: EdgeLatentGatedTransfer(d_edge) if d_edge is not None else None
                elif "base" in latrec_transfer:
                    transfer_fn = lambda: EdgeLatentBaseTransfer() if d_edge is not None else None
                else:
                    raise ValueError(
                        f"Unsupported strategy in latrec_transfer='{latrec_transfer}'. Expected 'base', 'gated', or 'lin_res'."
                    )

                is_e1e2 = any(loc in latrec_transfer for loc in ["e1e2", "e1_e2", "e1,e2", "e2"]) or latrec_transfer in ["edge_lr", "both", "edge_lr+identity"]
                self.latrec_e = transfer_fn()
                self.latrec_e_readout = transfer_fn() if is_e1e2 else None
            else:
                raise ValueError(
                    f"Unsupported latrec_transfer='{latrec_transfer}'. Supported choices: "
                    "['identity', 'node_lr_base', 'node_lr_lin_res', "
                    "'v_setmix', 'v_lin_res', 'nff_update', "
                    "'node_edge_e1_lin_res', 'edge_lr_e1_lin_res+node_lr_lin_res', "
                    "'edge_lr_e1_base', 'edge_lr_e1e2_base', "
                    "'edge_lr_e1_gated', 'edge_lr_e1e2_gated', "
                    "'edge_lr_e1_lin_res', 'edge_lr_e1e2_lin_res']."
                )
        else:
            self.latrec_s = None
            self.latrec_mlp = None
            self.latrec_e = None
            self.latrec_e_readout = None
            self.latrec_v = None

        if d_edge is not None or n_edge_types is not None:
            if None in [d_edge, n_edge_types]:
                raise ValueError("If either d_edge or n_edge_types are given both must be provided.")

            edge_in_feats = n_edge_types * 2 if self_cond else n_edge_types

            self.edge_in_proj = torch.nn.Sequential(
                torch.nn.Linear(edge_in_feats, d_edge), torch.nn.SiLU(inplace=False), torch.nn.Linear(d_edge, d_edge)
            )
            self.edge_out_proj = torch.nn.Sequential(
                torch.nn.Linear(d_edge, d_edge), torch.nn.SiLU(inplace=False), torch.nn.Linear(d_edge, n_edge_types)
            )

        in_feats = n_atom_feats + vocab_size if self_cond else n_atom_feats
        in_feats = in_feats + size_emb

        self.size_emb = torch.nn.Embedding(max_atoms, size_emb)
        self.feat_proj = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_model), torch.nn.SiLU(inplace=False), torch.nn.Linear(d_model, d_model)
        )

        self.dynamics = dynamics

        self.atom_classifier_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model), torch.nn.SiLU(inplace=False), torch.nn.Linear(d_model, vocab_size)
        )
        self.charge_classifier_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model), torch.nn.SiLU(inplace=False), torch.nn.Linear(d_model, 7)
        )

    def forward(
        self,
        coords,
        inv_feats,
        edge_feats=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
        atom_mask=None,
        prev_z=None,
        prev_node_latent=None,
        prev_edge_latent=None,
        prev_coord_latent=None,
    ):
        """Predict molecular coordinates and atom types

        Args:
            coords (torch.Tensor): Input coordinates, shape [batch_size, n_atoms, 3]
            inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, n_feats]
            edge_feats (torch.Tensor): In edge features, shape [batch_size, n_atoms, n_atoms, n_edge_types]
            cond_coords (torch.Tensor): Conditional coords, shape [batch_size, n_atoms, 3]
            cond_atomics (torch.Tensor): Conditional atom type logits, shape [batch_size, n_atoms, n_feats]
            cond_bonds (torch.Tensor): Cond bond type logits, shape [batch_size, n_atoms, n_atoms, n_edge_types]
            atom_mask (torch.Tensor): Mask for fake atoms, shape [batch_size, n_atoms], 1 for real atoms
            prev_z (dict, Optional): State dictionary {"s": node_latent, "e": edge_latent, "v": coord_latent} from previous pass
            prev_node_latent (torch.Tensor, Optional): Carried node scalar state from previous pass, shape [batch_size, n_atoms, d_model]
            prev_edge_latent (torch.Tensor, Optional): Carried edge state from previous pass, shape [batch_size, n_atoms, n_atoms, d_edge]
            prev_coord_latent (torch.Tensor, Optional): Carried coord set state from previous pass, shape [batch_size, n_sets, n_atoms, 3]

        Returns:
            (predicted coordinates, atom type logits, bond logits, atom charges, z)
        """

        if cond_coords is not None and not self.self_cond:
            raise ValueError("cond_coords was provided but the model was initialised with self_cond as False.")

        if cond_coords is None and self.self_cond:
            raise ValueError("The model was initialsed with self_cond but cond_coords was not provided.")

        if edge_feats is None and cond_bonds is not None:
            raise ValueError("edge_feats must be provided if using bond conditioning.")

        # Unpack carried latent states from prev_z if provided
        if prev_z is not None:
            if isinstance(prev_z, dict):
                if prev_node_latent is None:
                    prev_node_latent = prev_z.get("s")
                if prev_edge_latent is None:
                    prev_edge_latent = prev_z.get("e")
                if prev_coord_latent is None:
                    prev_coord_latent = prev_z.get("v")
            elif isinstance(prev_z, torch.Tensor):
                if prev_node_latent is None:
                    prev_node_latent = prev_z

        atom_mask = torch.ones_like(coords[..., 0]) if atom_mask is None else atom_mask
        pair_mask = (atom_mask.unsqueeze(2) * atom_mask.unsqueeze(1)).float()  # [B, N, N]
        adj_matrix = smolF.edges_from_nodes(coords, node_mask=atom_mask)

        # Embed the number of atoms in a mol into a small vector and concat this to inv feats for each atom
        n_atoms = atom_mask.sum(dim=-1, keepdim=True)
        assert (n_atoms < self.size_emb.num_embeddings).all(), "n_atoms exteed Embedding Range!"
        size_emb = self.size_emb(n_atoms).expand(-1, inv_feats.size(1), -1)
        inv_feats = torch.cat((inv_feats, size_emb), dim=-1)
        if cond_atomics is not None:
            inv_feats = torch.cat((inv_feats, cond_atomics), dim=-1)

        atom_feats = self.feat_proj(inv_feats)

        # --- latent recursion: node scalar states transfer ---
        if self.latent_recursion and prev_node_latent is not None and self.latrec_s is not None:
            atom_feats = self.latrec_s(atom_feats, prev_node_latent, atom_mask)
        atom_feats = atom_feats * atom_mask.unsqueeze(-1)
        # -----------------------------------------------------

        if edge_feats is not None:
            edge_feats = edge_feats.float()
            edge_feats = torch.cat((edge_feats, cond_bonds), dim=-1) if cond_bonds is not None else edge_feats
            edge_feats = self.edge_in_proj(edge_feats)
            # Injection Point 1 (E1/E2): inject after edge_in_proj
            if self.latrec_e is not None and prev_edge_latent is not None:
                edge_feats = self.latrec_e(edge_feats, prev_edge_latent, pair_mask)

        out = self.dynamics(
            coords,
            atom_feats,
            adj_matrix,
            atom_mask=atom_mask,
            edge_feats=edge_feats,
            cond_coords=cond_coords,
            prev_edge_latent=prev_edge_latent,
            latrec_e_readout=self.latrec_e_readout,
            pair_mask=pair_mask,
            latrec_v=self.latrec_v,
            prev_coord_latent=prev_coord_latent,
            prev_node_latent=prev_node_latent,
        )

        pred_edges = None
        v_latent = None
        if len(out) == 2:
            pred_coords, pred_feats = out
        elif len(out) == 3:
            if self.d_edge is None:
                pred_coords, pred_feats, v_latent = out
            else:
                pred_coords, pred_feats, pred_edges = out
        elif len(out) == 4:
            pred_coords, pred_feats, pred_edges, v_latent = out

        pred_coords = smolF.zero_com(pred_coords, node_mask=atom_mask)
        pred_coords = pred_coords * atom_mask.unsqueeze(-1)

        v_carried = v_latent if self.latent_recursion else None
        node_latent = pred_feats * atom_mask.unsqueeze(-1) if self.latent_recursion else None

        type_logits = self.atom_classifier_head(pred_feats)
        charge_logits = self.charge_classifier_head(pred_feats)

        # If we are predicting edges ensure that the matrix is symmetrical
        if pred_edges is not None:
            pred_edges = pred_edges + pred_edges.transpose(1, 2)
            edge_logits = self.edge_out_proj(pred_edges)
            e_latent = pred_edges * pair_mask.unsqueeze(-1) if self.latent_recursion else None
            z = {"s": node_latent, "e": e_latent, "v": v_carried} if self.latent_recursion else None
            return pred_coords, type_logits, edge_logits, charge_logits, z

        z = {"s": node_latent, "e": None, "v": v_carried} if self.latent_recursion else None
        return pred_coords, type_logits, charge_logits, z
