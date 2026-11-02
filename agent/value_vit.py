import torch
from torch import nn

from agent.noisy_nn import NoisyLinear
from agent.shared import *


class ValueViT(nn.Module):
    """
    Chess Visual Transformer for Value Estimation.
    - Input: Tensor of shape (B, 17, 8, 8)
        - Planes 0-5:   White pieces (P, N, B, R, Q, K)
        - Planes 6-11:  Black pieces (P, N, B, R, Q, K)
        - Plane 12:     Turn (1.0 for White, 0.0 for Black)
        - Plane 13:     White Queenside castling rights (1.0 or 0.0)
        - Plane 14:     White Kingside castling rights (1.0 or 0.0)
        - Plane 15:     Black Queenside castling rights (1.0 or 0.0)
        - Plane 16:     Black Kingside castling rights (1.0 or 0.0)
    - Output: Tensor of shape (B, 1) in range [-1.0, 1.0] (White-centric value)
    """
    def __init__(self, img_size=8, in_channels=17, patch_size=1, embed_dim=256,
        num_heads=8, depth=4, mlp_dim=512):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, (img_size // patch_size) ** 2)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mlp_value_func = NoisyLinear(embed_dim, 1)
        self.tanh = nn.Tanh()

    def reset_noise(self):
        """Resets exploration noise across all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x_0 = x[:, 0]  # CLS token
        v = self.mlp_value_func(x_0)
        return self.tanh(v)
