import torch
from torch import nn

from agent.shared import *


class ActorViT(nn.Module):
    """
    Chess Visual Transformer.
    - Input:
        - Planes 0-5:   White pieces (P, N, B, R, Q, K)
        - Planes 6-11:  Black pieces (P, N, B, R, Q, K)
        - Plane 12:     Turn (1.0 for White, 0.0 for Black)
        - Plane 13:     White Queenside castling rights (1.0 or 0.0)
        - Plane 14:     White Kingside castling rights (1.0 or 0.0)
        - Plane 15:     Black Queenside castling rights (1.0 or 0.0)
        - Plane 16:     Black Kingside castling rights (1.0 or 0.0)
    - Output: Tensor of shape=(4164,)
    """
    def __init__(self, img_size=8, in_channels=17, patch_size=1, embed_dim=768, 
        num_heads=8, depth=6, mlp_dim=1024, number_of_actions=4164):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, (img_size // patch_size) ** 2)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mlp_policy_func = nn.Linear(embed_dim, number_of_actions)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x_0 = x[:, 0] #x_0 = CLS
        pi = self.mlp_policy_func(x_0)
        return pi