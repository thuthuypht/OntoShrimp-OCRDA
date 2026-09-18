from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny
from torchvision import transforms

CLASS_NAMES=["BG","Healthy","WSSV"]

class VisualModel(nn.Module):
    """Inference-only copy of the frozen OCRDA-v7.4.4 visual network."""
    def __init__(self, emb_dim=256, dropout=0.20):
        super().__init__()
        # pretrained=False / weights=None: exact checkpoint weights are loaded immediately.
        net=convnext_tiny(weights=None)
        in_dim=net.classifier[2].in_features
        net.classifier[2]=nn.Identity()
        self.backbone=net
        self.proj=nn.Sequential(
            nn.Linear(in_dim,512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512,emb_dim), nn.LayerNorm(emb_dim),
        )
        self.source_head=nn.Linear(emb_dim,len(CLASS_NAMES))
        self.target_head=nn.Linear(emb_dim,len(CLASS_NAMES))
    def encode(self,x):
        return F.normalize(self.proj(self.backbone(x)),dim=1)
    def forward(self,x,head="target"):
        z=self.encode(x)
        if head=="source": return z,self.source_head(z)
        if head=="target": return z,self.target_head(z)
        raise ValueError(head)

def test_transform(image_size=224):
    norm=transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    return transforms.Compose([
        transforms.Resize(int(image_size*1.12)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(), norm,
    ])
