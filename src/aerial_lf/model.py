"""Behavior-preserving public definition of the archived F15 model."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 36)
        )

    def forward(self, scene_tokens):
        z = self.net(scene_tokens.mean(1)).view(-1, 3, 6, 2)
        first = F.softplus(z[..., 0:1, 0])
        increments = F.softplus(z[..., 1:, 0])
        forward = torch.cat((first, first + torch.cumsum(increments, -1)), -1)
        return torch.stack((forward, z[..., 1]), -1)


class DINOResidual(nn.Module):
    def __init__(self):
        super().__init__()
        rank = 32
        self.adapter = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(), nn.LayerNorm(256)
        )
        self.anchor_projection = nn.Sequential(
            nn.Linear(2, 256), nn.GELU(), nn.Linear(256, 256)
        )
        self.query, self.key, self.value = (
            nn.Linear(256, rank, bias=False) for _ in range(3)
        )
        self.output = nn.Linear(rank, 256)
        self.bound = 0.12
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, scene_tokens, uav_tokens, valid_mask, anchors_ego_m, pnp_gate):
        tokens = self.adapter(uav_tokens) + self.anchor_projection(anchors_ego_m)
        tokens = tokens.masked_fill(~valid_mask[..., None], 0)
        q, k, v = self.query(scene_tokens), self.key(tokens), self.value(tokens)
        logits = (q @ k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        context = logits.masked_fill(~valid_mask[:, None, :], -1e4).softmax(-1) @ v
        residual = self.bound * torch.tanh(self.output(context) / self.bound)
        return scene_tokens + pnp_gate[:, None, None] * residual


class DenseSemanticResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.token_projection = nn.Sequential(
            nn.LayerNorm(96), nn.Linear(96, 256), nn.GELU()
        )
        self.position_projection = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 256)
        )
        rank = 32
        self.query, self.key, self.value = (
            nn.Linear(256, rank, bias=False) for _ in range(3)
        )
        self.output = nn.Linear(rank, 256)
        self.bound = 0.12

    @staticmethod
    def pool_positions(grid_body_xy_m, output_size):
        return F.adaptive_avg_pool2d(
            grid_body_xy_m.permute(0, 3, 1, 2), output_size
        ).permute(0, 2, 3, 1)

    def forward(
        self,
        scene_tokens,
        semantic_map,
        grid_body_xy_m,
        invalid,
        unknown,
        pnp_gate,
    ):
        feature = self.encoder(semantic_map)
        batch, _, height, width = feature.shape
        tokens = self.token_projection(feature.flatten(2).transpose(1, 2))
        positions = self.pool_positions(grid_body_xy_m, (height, width))
        tokens = tokens + self.position_projection(positions.flatten(1, 2))
        known = ~(invalid.bool() | unknown.bool())
        coverage = F.adaptive_avg_pool2d(
            known.float().unsqueeze(1), (height, width)
        ).flatten(1)
        token_mask = coverage >= 0.5
        q, k, v = self.query(scene_tokens), self.key(tokens), self.value(tokens)
        logits = (q @ k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        context = logits.masked_fill(~token_mask[:, None, :], -1e4).softmax(-1) @ v
        context = context * token_mask.any(-1)[:, None, None]
        residual = self.bound * torch.tanh(self.output(context) / self.bound)
        return scene_tokens + pnp_gate[:, None, None] * residual, token_mask


class F15JointHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = Head()
        self.dino = DINOResidual()
        self.dense_semantic = DenseSemanticResidual()

    def forward(
        self,
        scene_tokens,
        uav_tokens,
        uav_valid_mask,
        uav_anchors_ego_m,
        semantic_map,
        grid_body_xy_m,
        invalid,
        unknown,
        pnp_gate,
    ):
        scene_tokens = self.dino(
            scene_tokens,
            uav_tokens,
            uav_valid_mask,
            uav_anchors_ego_m,
            pnp_gate,
        )
        scene_tokens, token_mask = self.dense_semantic(
            scene_tokens,
            semantic_map,
            grid_body_xy_m,
            invalid,
            unknown,
            pnp_gate,
        )
        return self.head(scene_tokens), {"semantic_token_mask": token_mask}


AerialLatentFusion = F15JointHeadModel


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

