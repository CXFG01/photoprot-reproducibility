"""Phase 1 model: a deliberately boring ResNet with two classification heads.

The point of Phase 1 is to find out whether a single 2D projection carries CATH
architecture signal at all, not to win an architecture search. The backbone also
has to hand back its penultimate embedding, because the cheapest useful next
experiment is to ask whether nearest neighbours in that space are structurally
close - before building anything contrastive.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
}


class MultiHeadResNet(nn.Module):
    """Shared trunk, one linear head per CATH level.

    Only class (5) and architecture (43) get heads. Topology (1,471 labels,
    median 3 domains each) and superfamily (6,576 labels, median 1) are too
    sparse to classify, and under a superfamily-held-out split every test
    superfamily is unseen by construction. Those levels are evaluated by
    retrieval instead.
    """

    def __init__(
        self,
        n_class: int,
        n_arch: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone}")
        ctor, weights, feat_dim = BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        self.feat_dim = feat_dim
        net.fc = nn.Identity()
        self.trunk = net
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head_C = nn.Linear(feat_dim, n_class)
        self.head_A = nn.Linear(feat_dim, n_arch)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        z = self.trunk(x)
        h = self.dropout(z)
        logits = {"C": self.head_C(h), "A": self.head_A(h)}
        if return_embedding:
            return logits, z
        return logits

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalised penultimate features, for the retrieval probe."""
        z = self.trunk(x)
        return torch.nn.functional.normalize(z, dim=1)
