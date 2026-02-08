import torch
import torch.nn as nn
from transformer_encoder import Encoder  # Ensure these files are in your repo
from transformer_decoder import Decoder  # Ensure these files are in your repo

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TemporalAttention(nn.Module):
    """
    Temporal Attention Mechanism with Sparse Mask support.
    Focuses on local neighborhood within a temporal window.
    """

    def __init__(self, dim, heads=2, window_size=1, local_window_radius=5,
                 qkv_bias=False, qk_scale=None, dropout=0., causal=True):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."
        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size
        self.local_window_radius = local_window_radius

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # Register sparse attention mask
        self.register_buffer('sparse_mask', self.generate_sparse_mask(window_size, local_window_radius, causal))

    def generate_sparse_mask(self, T, local_window_radius, causal):
        """Generates a mask to restrict attention to a local temporal window."""
        mask = torch.full((T, T), float("-inf"))
        for i in range(T):
            if causal:
                # Causal: attend only to current and past steps within radius
                start = max(0, i - local_window_radius)
                end = i + 1
            else:
                # Non-causal: attend to past and future steps within radius
                start = max(0, i - local_window_radius)
                end = min(T, i + local_window_radius + 1)
            mask[i, start:end] = 0.0
        return mask

    def forward(self, x):
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)

        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Apply sparse mask
        attn = attn + self.sparse_mask.unsqueeze(0).unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:
            x = x.reshape(B_prev, T_prev, C_prev)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SCT_MSA(nn.Module):
    """
    Sparse Causal Temporal Multi-Head Self-Attention (SCT_MSA).
    Renamed from CT_MSA to match the manuscript.
    """

    def __init__(self, dim, depth, heads, window_size, mlp_dim, num_time, dropout=0., local_window_radius=5):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=dim, heads=heads, window_size=window_size,
                                  local_window_radius=local_window_radius,
                                  dropout=dropout, causal=True),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        # x shape: [B, embed_size, K, seq_len]
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * n, t, c)
        x = x + self.pos_embedding
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2)
        return x


class TrajectoryModel(nn.Module):
    """
    Trajectory Prediction Model using Parallel Fusion (SCT_MSA + Encoder)
    and Feature Aggregation.
    """

    def __init__(self, in_size, obs_len, pred_len, embed_size, enc_num_layers,
                 int_num_layers_list, heads, forward_expansion, local_window_radius=4):
        super(TrajectoryModel, self).__init__()

        self.seq_len = obs_len + pred_len
        self.embed_size_per_step = 64

        # SCT_MSA Path
        self.embedding_sct = nn.Linear(2, self.embed_size_per_step)
        self.sct_msa = SCT_MSA(
            dim=self.embed_size_per_step,
            depth=enc_num_layers,
            heads=heads,
            window_size=self.seq_len,
            mlp_dim=self.embed_size_per_step * forward_expansion,
            num_time=self.seq_len,
            dropout=0.,
            local_window_radius=local_window_radius
        ).to(device)
        self.sct_out_proj = nn.Linear(self.embed_size_per_step, embed_size)

        # Original Encoder Path
        self.embedding_enc = nn.Linear(in_size * self.seq_len, embed_size)
        self.mode_encoder = Encoder(embed_size, enc_num_layers, heads, forward_expansion, islinear=True)

        # Heads and Decoder
        self.cls_head = nn.Linear(embed_size, 1)  # Classification head for mode scores
        self.nei_embedding = nn.Linear(in_size * obs_len, embed_size)
        self.social_decoder = Decoder(embed_size, int_num_layers_list[1], heads, forward_expansion, islinear=False)
        self.reg_head = nn.Linear(embed_size, in_size * pred_len)  # Regression head for trajectories

    def spatial_interaction(self, ped, neis, mask):
        """Handles spatial interaction between pedestrians and neighbors."""
        neis = neis.reshape(neis.shape[0], neis.shape[1], -1)  # [B, N, obs_len*2]
        nei_embeddings = self.nei_embedding(neis)
        mask = mask[:, 0:1].repeat(1, ped.shape[1], 1)
        int_feat = self.social_decoder(ped, nei_embeddings, mask)
        return int_feat

    def forward(self, ped_obs, neis_obs, motion_modes, mask, closest_mode_indices, test=False, num_k=20):
        """
        Forward pass.

        Args:
            ped_obs: [B, obs_len, 2] - Observed trajectory
            neis_obs: [B, N, obs_len, 2] - Neighbors' observed trajectories
            motion_modes: [K, pred_len, 2] - Motion modes anchors
            mask: [B, N, N] - Attention mask
            closest_mode_indices: [B] - Ground truth mode indices
            test: bool - Inference mode flag
            num_k: int - Number of top-k modes to predict in test mode

        Returns:
            Train mode: pred_traj [B, pred_len*2], scores [B, K]
            Test mode: pred_trajs [B, num_k, pred_len*2], scores [B, K]
        """
        # Expand and concatenate
        ped_obs = ped_obs.unsqueeze(1).repeat(1, motion_modes.shape[0], 1, 1)
        motion_modes = motion_modes.unsqueeze(0).repeat(ped_obs.shape[0], 1, 1, 1)
        ped_seq = torch.cat((ped_obs, motion_modes), dim=-2)  # [B, K, seq_len, 2]

        # SCT_MSA Path
        ped_embedding_sct = self.embedding_sct(ped_seq)
        ped_embedding_sct = ped_embedding_sct.permute(0, 3, 1, 2)
        sct_feat = self.sct_msa(ped_embedding_sct)
        sct_feat = sct_feat.mean(dim=-1)
        sct_feat = sct_feat.permute(0, 2, 1)
        sct_feat = self.sct_out_proj(sct_feat)

        # Original Encoder Path
        ped_seq_flat = ped_seq.reshape(ped_seq.shape[0], ped_seq.shape[1], -1)
        ped_embedding_enc = self.embedding_enc(ped_seq_flat)
        enc_feat = self.mode_encoder(ped_embedding_enc)

        # Fusion (Summation)
        ped_feat = sct_feat + enc_feat
        scores = self.cls_head(ped_feat).squeeze()

        # Training Mode
        if not test:
            index1 = torch.arange(closest_mode_indices.shape[0]).to(device)
            index2 = closest_mode_indices
            closest_feat = ped_feat[index1, index2].unsqueeze(1)
            int_feat = self.spatial_interaction(closest_feat, neis_obs, mask)
            pred_traj = self.reg_head(int_feat.squeeze())
            return pred_traj, scores

        # Test Mode
        if test:
            top_k_indices = torch.topk(scores, k=num_k, dim=-1).indices
            top_k_indices = top_k_indices.flatten()
            index1 = torch.arange(ped_feat.shape[0]).to(device).unsqueeze(1).repeat(1, num_k).flatten()
            index2 = top_k_indices
            top_k_feat = ped_feat[index1, index2].reshape(ped_feat.shape[0], num_k, -1)
            int_feats = self.spatial_interaction(top_k_feat, neis_obs, mask)
            pred_trajs = self.reg_head(int_feats)
            return pred_trajs, scores


# Example usage for verification
if __name__ == "__main__":
    print(f"Initializing model on {device}...")
    model = TrajectoryModel(
        in_size=2,
        obs_len=8,
        pred_len=12,
        embed_size=64,
        enc_num_layers=2,
        int_num_layers_list=[2, 2],
        heads=4,
        forward_expansion=4
    ).to(device)

    # Dummy Data
    B, N, K = 16, 10, 20
    ped_obs = torch.randn(B, 8, 2).to(device)
    neis_obs = torch.randn(B, N, 8, 2).to(device)
    motion_modes = torch.randn(K, 12, 2).to(device)
    mask = torch.ones(B, N, N).to(device)
    closest_mode_indices = torch.randint(0, K, (B,)).to(device)

    # Train mode check
    pred_traj, scores = model(ped_obs, neis_obs, motion_modes, mask, closest_mode_indices, test=False)
    print(f"Train Mode - Output Shapes: Traj {pred_traj.shape}, Scores {scores.shape}")

    # Test mode check
    pred_trajs, scores = model(ped_obs, neis_obs, motion_modes, mask, closest_mode_indices, test=True, num_k=5)
    print(f"Test Mode - Output Shapes: Traj {pred_trajs.shape}, Scores {scores.shape}")
