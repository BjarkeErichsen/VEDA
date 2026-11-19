import torch
import torch.nn as nn


# =============================================================================
# Node-Level Latent Recursion Mechanisms (Scalar State Transfer)
# =============================================================================

class NodeLatentBaseTransfer(nn.Module):
    """Direct addition: node = node + prev_node."""

    def forward(self, atom_feats, prev_node_latent, atom_mask=None):
        mask = atom_mask.unsqueeze(-1).float() if atom_mask is not None else 1.0
        return atom_feats + prev_node_latent * mask


NodeLatentIdentityTransfer = NodeLatentBaseTransfer


class NodeLatentLinResTransfer(nn.Module):
    """Linear residual transfer: node = node + Linear([node, node - prev_node])."""

    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(2 * d_model, d_model, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.linear.weight._no_reinit = True
        self.linear.bias._no_reinit = True

    def forward(self, atom_feats, prev_node_latent, atom_mask=None):
        r = atom_feats - prev_node_latent
        concat = torch.cat([atom_feats, r], dim=-1)
        mask = atom_mask.unsqueeze(-1).float() if atom_mask is not None else 1.0
        return atom_feats + self.linear(concat) * mask


# Alias for backward compatibility
NodeLatentTransfer = NodeLatentIdentityTransfer


# =============================================================================
# Coordinate-Level Latent Recursion Mechanisms (Spatial Vector Set Transfer)
# =============================================================================

class CoordLatentSetMixTransfer(nn.Module):
    """v_setmix: v = v + g * W * CoordNorm(v_prev)
    
    Mixing across full 64 coordinate sets without spatial bias or non-linearity.
    Zeroed at initialization by scalar gate g (~4k params for 64 sets).
    """

    def __init__(self, n_coord_sets=64, coord_norm="length", eps=1e-6):
        super().__init__()
        from models.semla import CoordNorm
        self.coord_norm = CoordNorm(n_coord_sets, norm=coord_norm, eps=eps)
        self.proj = nn.Linear(n_coord_sets, n_coord_sets, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))
        self.gate._no_reinit = True

    def forward(self, coords, prev_coords, node_mask, inv_feats=None, prev_inv_feats=None):
        if node_mask is not None and node_mask.dim() == 2:
            node_mask = node_mask.unsqueeze(1).expand(-1, coords.size(1), -1)
        v_norm = self.coord_norm(prev_coords, node_mask)
        # Linear mix across coordinate sets dimension S (dim 1): [B, S, N, 3] -> [B, N, 3, S] -> [B, S, N, 3]
        update = self.proj(v_norm.movedim(1, -1)).movedim(-1, 1)
        mask = node_mask.unsqueeze(-1).float() if node_mask is not None else 1.0
        v_out = coords + self.gate * update * mask
        return v_out, inv_feats


VSetMixTransfer = CoordLatentSetMixTransfer


class CoordLatentLinResTransfer(nn.Module):
    """v_lin_res: v = v + W([v, v - v_prev]) concatenated on set dimension S
    
    Linear mapping from 2*S to S across sets without spatial bias or non-linearity.
    Zeroed at initialization by weight matrix W (~8k params for 64 sets).
    """

    def __init__(self, n_coord_sets=64):
        super().__init__()
        self.linear = nn.Linear(2 * n_coord_sets, n_coord_sets, bias=False)
        nn.init.zeros_(self.linear.weight)
        self.linear.weight._no_reinit = True

    def forward(self, coords, prev_coords, node_mask, inv_feats=None, prev_inv_feats=None):
        if node_mask is not None and node_mask.dim() == 2:
            node_mask = node_mask.unsqueeze(1).expand(-1, coords.size(1), -1)
        r = coords - prev_coords
        concat = torch.cat([coords, r], dim=1)  # [B, 2*S, N, 3]
        update = self.linear(concat.movedim(1, -1)).movedim(-1, 1)
        mask = node_mask.unsqueeze(-1).float() if node_mask is not None else 1.0
        v_out = coords + update * mask
        return v_out, inv_feats


VLinResTransfer = CoordLatentLinResTransfer


class CoordNodeLatentNFFTransfer(nn.Module):
    """nff_update: v, s += NodeFeedForward(v_prev, s_prev)
    
    Interprets carried (v_prev, s_prev) jointly through NodeFeedForward block.
    Zeroed at initialization by two scalar gates gate_v and gate_s (~320k params with d_ff=d_model).
    """

    def __init__(self, d_model, n_coord_sets=64, coord_norm="length"):
        super().__init__()
        from models.semla import NodeFeedForward
        self.nff = NodeFeedForward(d_model, n_coord_sets, d_ff=d_model, coord_norm=coord_norm)
        self.gate_v = nn.Parameter(torch.zeros(1))
        self.gate_s = nn.Parameter(torch.zeros(1))
        self.gate_v._no_reinit = True
        self.gate_s._no_reinit = True

    def forward(self, coords, prev_coords, node_mask, inv_feats=None, prev_inv_feats=None):
        if node_mask is not None and node_mask.dim() == 2:
            node_mask = node_mask.unsqueeze(1).expand(-1, coords.size(1), -1)
        s_in = prev_inv_feats if prev_inv_feats is not None else inv_feats
        delta_v, delta_s = self.nff(prev_coords, s_in, node_mask)
        v_mask = node_mask.unsqueeze(-1).float() if node_mask is not None else 1.0
        s_mask = (node_mask[:, 0, :].unsqueeze(-1).float() if node_mask.dim() == 3 else node_mask.unsqueeze(-1).float()) if node_mask is not None else 1.0
        
        v_out = coords + self.gate_v * delta_v * v_mask
        s_out = inv_feats + self.gate_s * delta_s * s_mask if inv_feats is not None else inv_feats
        return v_out, s_out


NFFUpdateTransfer = CoordNodeLatentNFFTransfer


# =============================================================================
# Edge-Level Latent Recursion Mechanisms (Pairwise State Transfer)
# =============================================================================

class EdgeLatentBaseTransfer(nn.Module):
    """Direct addition: edge = edge + prev_edge."""

    def forward(self, edge_feats, prev_edge_latent, pair_mask=None):
        mask = pair_mask.unsqueeze(-1).float() if pair_mask is not None else 1.0
        return edge_feats + prev_edge_latent * mask


class EdgeLatentGatedTransfer(nn.Module):
    """Gated projection: edge = edge + gate * Linear(LayerNorm(prev_edge))."""

    def __init__(self, d_edge):
        super().__init__()
        self.norm = nn.LayerNorm(d_edge)
        self.proj = nn.Linear(d_edge, d_edge)
        self.gate = nn.Parameter(torch.zeros(1))
        self.gate._no_reinit = True

    def forward(self, edge_feats, prev_edge_latent, pair_mask=None):
        mask = pair_mask.unsqueeze(-1).float() if pair_mask is not None else 1.0
        update = self.proj(self.norm(prev_edge_latent)) * mask
        return edge_feats + self.gate * update


class EdgeLatentLinResTransfer(nn.Module):
    """Linear residual transfer: edge = edge + Linear([edge, edge - prev_edge])."""

    def __init__(self, d_edge):
        super().__init__()
        self.linear = nn.Linear(2 * d_edge, d_edge, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.linear.weight._no_reinit = True
        self.linear.bias._no_reinit = True

    def forward(self, edge_feats, prev_edge_latent, pair_mask=None):
        r = edge_feats - prev_edge_latent
        concat = torch.cat([edge_feats, r], dim=-1)
        mask = pair_mask.unsqueeze(-1).float() if pair_mask is not None else 1.0
        return edge_feats + self.linear(concat) * mask


# Alias for backward compatibility if referenced
EdgeLatentTransfer = EdgeLatentGatedTransfer


__all__ = [
    "NodeLatentBaseTransfer",
    "NodeLatentIdentityTransfer",
    "NodeLatentLinResTransfer",
    "NodeLatentTransfer",
    "CoordLatentSetMixTransfer",
    "VSetMixTransfer",
    "CoordLatentLinResTransfer",
    "VLinResTransfer",
    "CoordNodeLatentNFFTransfer",
    "NFFUpdateTransfer",
    "EdgeLatentBaseTransfer",
    "EdgeLatentGatedTransfer",
    "EdgeLatentLinResTransfer",
    "EdgeLatentTransfer",
]

