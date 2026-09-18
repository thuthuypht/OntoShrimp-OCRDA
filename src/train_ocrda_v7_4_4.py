#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ocrda_v7_4_4.py

Reference implementation of OCRDA-v7.4.4:
  - ConvNeXt-Tiny visual backbone with LayerNorm projection
  - relation-aware R-GCN ontology encoder with recognition-specific topology
  - frozen source teacher + EMA target teacher
  - ontology tri-consensus target selection
  - prior-aware teacher pseudo-label curriculum
  - recognition-only ontology-smoothed soft semantic pseudo-label learning
  - ontology-conditioned prototype contrastive learning
  - class-conditional cosine mean alignment (no covariance matching)
  - EMA target-prototype memory
  - ontology-ranking topology regularization
  - robust multi-evidence label-shift target-prior estimation (no target labels)
  - selected-distribution-aware continuous adaptation gate
  - tempered pseudo-label class quota decoupled from the estimated target prior
  - hard selected-distribution anti-collapse gate with safe-state rollback
  - label-free safe-checkpoint weight averaging (SWA-style ensemble)
  - actual rolling plateau/drift controller for mature adaptation
  - source-retention trust region with causal next-epoch control
  - reproducible multi-seed evaluation

Development protocol:
  Source train/validation/test: SDBD
  Target adaptation: TSBD target_adapt (labels hidden)
  Target development diagnostics: exposed TSBD development partition
  Final confirmatory target test: intentionally NOT used during method development
  Classes: BG, Healthy, WSSV

Protocol safeguards:
  * target_adapt_final.csv labels are NEVER read for training.
  * a target final test is evaluated only when --evaluate-final-test is explicitly supplied.
  * ontology properties with alignmentEligible=false are excluded from the OCRDA graph.
  * provenance edges (hasDataset, hasLearningDomain, hasSpecies, etc.) are therefore unavailable
    to the adaptation encoder, preventing ontology-side dataset-identity leakage.

This script establishes the image-classification OCRDA branch. The same R-GCN
encoder and OCRDA losses can later be attached to lesion/ROI features from the
selected detector backbone.

Dependencies:
  torch torchvision pandas pillow rdflib scikit-learn numpy
"""

import argparse
import copy
import json
import math
import os
import random
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, BNode
from rdflib.util import guess_format
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    matthews_corrcoef, precision_recall_fscore_support, confusion_matrix
)

BASE = "http://www.nhatrang.edu.vn/ontology/shrimp-disease#"
SD = Namespace(BASE)

CLASS_NAMES = ["BG", "Healthy", "WSSV"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}
ANCHOR_CLASS_LOCAL = {
    "BG": "BlackGillRecognitionState",
    "Healthy": "HealthyRecognitionState",
    "WSSV": "WSSVRecognitionState",
}

# OCRDA-v7.4.4 separates contextual/reasoning relations from visual-recognition
# similarity. COOCCURSWITH and HASPATHOGEN remain available to the relation-aware
# R-GCN, but they are intentionally excluded from class-label smoothing/topology.
RECOGNITION_TOPOLOGY_RELATIONS = {
    "IS_A",
    "SYMPTOMSUPPORTSDISEASE",
    "TYPICALLOCATION",
}
RECOGNITION_TOPOLOGY_TAU = 1.50
RECOGNITION_TOPOLOGY_MAX_OFFDIAG = 0.35


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def normalize_label(row):
    for col in ("disease_label_common", "disease_label_raw", "label"):
        if col in row.index and pd.notna(row[col]):
            v = str(row[col]).strip()
            if v in CLASS_TO_ID:
                return v
    raise ValueError(f"Cannot map row to {CLASS_NAMES}: {row.to_dict()}")


class ShrimpCSVDataset(Dataset):
    def __init__(self, csv_path, sdbd_root, tsbd_root, transform, expose_labels=True):
        self.df = pd.read_csv(csv_path)
        self.sdbd_root = Path(sdbd_root)
        self.tsbd_root = Path(tsbd_root)
        self.transform = transform
        self.expose_labels = expose_labels

        if expose_labels:
            keep = []
            for _, row in self.df.iterrows():
                try:
                    normalize_label(row)
                    keep.append(True)
                except Exception:
                    keep.append(False)
            self.df = self.df.loc[keep].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _image_path(self, row):
        ds = str(row.get("dataset_id", "")).strip()
        rel = str(row.get("original_relpath", row.get("filename", "")))
        rel = rel.replace("\\", os.sep).replace("/", os.sep)
        if ds == "SDBD":
            root = self.sdbd_root
        elif ds == "TSBD":
            root = self.tsbd_root
        else:
            raise ValueError(f"Unknown dataset_id={ds}")
        p = root / rel
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        return p

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = self._image_path(row)
        with Image.open(p) as im:
            x = self.transform(im.convert("RGB"))
        if self.expose_labels:
            y = CLASS_TO_ID[normalize_label(row)]
        else:
            # Strong safeguard: labels in target-adaptation CSV are never exposed.
            y = -1
        sid = str(row.get("sample_id", p.name))
        return x, y, sid



def build_transforms(image_size):
    """
    OCRDA-v7 uses asymmetric target views:
      - source_train_tf: moderate augmentation for supervised source learning;
      - target_weak_tf: weak view for the EMA teacher;
      - target_strong_tf: strong view for student consistency/pseudo-label learning;
      - test_tf: deterministic evaluation.

    ConvNeXt uses ImageNet normalization, but LayerNorm in the backbone is less
    sensitive to source/target batch-statistics than BatchNorm-based backbones.
    """
    norm = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )

    source_train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.20, contrast=0.20, saturation=0.15, hue=0.03
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
            p=0.15,
        ),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(
            p=0.15, scale=(0.02, 0.12), ratio=(0.5, 2.0), value="random"
        ),
    ])

    target_weak_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.10)),
        transforms.RandomCrop(image_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])

    target_strong_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.20, hue=0.04
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
            p=0.25,
        ),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(
            p=0.25, scale=(0.02, 0.18), ratio=(0.4, 2.5), value="random"
        ),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.12)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        norm,
    ])
    return source_train_tf, target_weak_tf, target_strong_tf, test_tf


class ShrimpDualViewCSVDataset(ShrimpCSVDataset):
    """
    Unlabeled target dataset that returns weak/strong stochastic views of the
    same image. Labels in the CSV remain hidden even when present physically.
    """
    def __init__(self, csv_path, sdbd_root, tsbd_root, weak_transform, strong_transform):
        super().__init__(
            csv_path=csv_path,
            sdbd_root=sdbd_root,
            tsbd_root=tsbd_root,
            transform=None,
            expose_labels=False,
        )
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = self._image_path(row)
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            xw = self.weak_transform(rgb)
            xs = self.strong_transform(rgb)
        sid = str(row.get("sample_id", p.name))
        return xw, xs, -1, sid


def class_balancing(dataset):
    labels = [CLASS_TO_ID[normalize_label(dataset.df.iloc[i])] for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(float)
    inv = 1.0 / np.maximum(counts, 1.0)
    sample_weights = [inv[y] for y in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    ce_weights = len(labels) / (len(CLASS_NAMES) * np.maximum(counts, 1.0))
    return sampler, torch.tensor(ce_weights, dtype=torch.float32)


# ---------------------------------------------------------------------
# Ontology graph projection
# ---------------------------------------------------------------------
def local_name(uri):
    s = str(uri)
    return s.split("#")[-1] if "#" in s else s.rsplit("/", 1)[-1]


def literal_bool(v):
    return str(v).strip().lower() in {"true", "1"}


def class_depths(g):
    classes = set(g.subjects(RDF.type, OWL.Class))
    parents = defaultdict(list)
    for c, _, p in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(c, URIRef) and isinstance(p, URIRef) and c in classes and p in classes:
            parents[c].append(p)

    memo = {}
    def depth(c, trail=None):
        if c in memo:
            return memo[c]
        trail = set() if trail is None else trail
        if c in trail:
            return 0
        trail.add(c)
        ps = parents.get(c, [])
        d = 0 if not ps else 1 + max(depth(p, trail.copy()) for p in ps)
        memo[c] = d
        return d

    return {c: depth(c) for c in classes}


def semantic_anchor_fillers(g, state):
    fillers = []
    for r in g.objects(state, RDFS.subClassOf):
        if not isinstance(r, BNode) or (r, RDF.type, OWL.Restriction) not in g:
            continue
        if next(g.objects(r, OWL.onProperty), None) != SD.hasSemanticAnchor:
            continue
        for pred in (OWL.someValuesFrom, OWL.allValuesFrom, OWL.hasValue):
            f = next(g.objects(r, pred), None)
            if isinstance(f, URIRef):
                fillers.append(f)
                break
    return fillers


def compute_specificity(g):
    depths = class_depths(g)
    vals = []
    for c in CLASS_NAMES:
        state = SD[ANCHOR_CLASS_LOCAL[c]]
        fs = semantic_anchor_fillers(g, state)
        ds = [depths.get(f, 0) for f in fs]
        vals.append(float(max(ds) if ds else depths.get(state, 0)))
    mx = max(vals) if vals else 1.0
    return np.array([v / max(mx, 1.0) for v in vals], dtype=np.float32)


def compute_anchor_topology(
    g,
    semantic_base_edges,
    tau_d=RECOGNITION_TOPOLOGY_TAU,
    allowed_relations=None,
    max_offdiag=RECOGNITION_TOPOLOGY_MAX_OFFDIAG,
):
    """
    Recognition-specific class similarity between semantic ANCHOR FILLER SETS.

    OCRDA-v7.4.4 deliberately excludes contextual relations such as
    COOCCURSWITH and HASPATHOGEN from *recognition* similarity, because semantic
    relatedness/co-occurrence must not imply visual-class similarity.  The full
    eligible semantic graph is still passed to the R-GCN for reasoning.
    """
    if allowed_relations is None:
        allowed_relations = RECOGNITION_TOPOLOGY_RELATIONS
    allowed_relations = set(allowed_relations)

    blocked_states = {SD[ANCHOR_CLASS_LOCAL[c]] for c in CLASS_NAMES}
    und = defaultdict(set)

    for s, r, o in semantic_base_edges:
        if s in blocked_states or o in blocked_states:
            continue
        if r == "HASSEMANTICANCHOR":
            continue
        if r not in allowed_relations:
            continue
        und[s].add(o)
        und[o].add(s)

    def shortest_between_sets(A, B):
        A = [a for a in A if isinstance(a, URIRef)]
        B = set(B)
        q = deque((a, 0) for a in A)
        seen = set(A)
        while q:
            u, d = q.popleft()
            if u in B:
                return d
            for v in und[u]:
                if v not in seen:
                    seen.add(v)
                    q.append((v, d + 1))
        return None

    sets = [semantic_anchor_fillers(g, SD[ANCHOR_CLASS_LOCAL[c]]) for c in CLASS_NAMES]
    finite = []
    dist = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.float32)

    for i in range(len(CLASS_NAMES)):
        for j in range(i + 1, len(CLASS_NAMES)):
            d = shortest_between_sets(sets[i], sets[j])
            if d is not None:
                finite.append(d)
            dist[i, j] = dist[j, i] = -1 if d is None else d

    fallback = (max(finite) + 2) if finite else 6
    dist[dist < 0] = fallback
    sim = np.exp(-dist / max(float(tau_d), 1e-6)).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    if max_offdiag is not None:
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                if i != j:
                    sim[i, j] = min(float(sim[i, j]), float(max_offdiag))
    return sim


def compute_full_context_topology(g, semantic_base_edges, tau_d=2.0):
    """Diagnostic only: v7.4.2-style topology using all eligible relations."""
    blocked_states = {SD[ANCHOR_CLASS_LOCAL[c]] for c in CLASS_NAMES}
    und = defaultdict(set)
    for s, r, o in semantic_base_edges:
        if s in blocked_states or o in blocked_states or r == "HASSEMANTICANCHOR":
            continue
        und[s].add(o); und[o].add(s)
    sets = [semantic_anchor_fillers(g, SD[ANCHOR_CLASS_LOCAL[c]]) for c in CLASS_NAMES]
    def shortest(A,B):
        B=set(B); q=deque((a,0) for a in A); seen=set(A)
        while q:
            u,d=q.popleft()
            if u in B: return d
            for v in und[u]:
                if v not in seen: seen.add(v); q.append((v,d+1))
        return None
    finite=[]; dist=np.zeros((len(CLASS_NAMES),len(CLASS_NAMES)),dtype=np.float32)
    for i in range(len(CLASS_NAMES)):
        for j in range(i+1,len(CLASS_NAMES)):
            d=shortest(sets[i],sets[j])
            if d is not None: finite.append(d)
            dist[i,j]=dist[j,i]=-1 if d is None else d
    fallback=(max(finite)+2) if finite else 6
    dist[dist<0]=fallback
    sim=np.exp(-dist/max(float(tau_d),1e-6)).astype(np.float32)
    np.fill_diagonal(sim,1.0)
    return sim


def load_local_rdf_graph(path_text):
    """Windows-safe local RDF/OWL loader."""
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Ontology file not found: {path}")
    path = path.resolve()

    suffix = path.suffix.lower()
    if suffix in {".owl", ".rdf", ".xml"}:
        fmt = "xml"
    else:
        fmt = guess_format(path.name) or "xml"

    g = Graph()
    with path.open("rb") as fh:
        g.parse(source=fh, format=fmt)
    return g


def ontology_semantic_graph(owl_path):
    g = load_local_rdf_graph(owl_path)

    object_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
    eligible_props = set()
    for p in object_props:
        role = {str(v) for v in g.objects(p, SD.alignmentRole)}
        eligible = any(literal_bool(v) for v in g.objects(p, SD.alignmentEligible))
        if "semantic" in role and eligible:
            eligible_props.add(p)

    nodes = {s for s in g.subjects(RDF.type, OWL.Class)
             if isinstance(s, URIRef) and str(s).startswith(BASE)}

    # Named individuals can be semantic fillers of hasValue restrictions.
    non_individual_types = {
        OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty,
        OWL.AnnotationProperty, OWL.Ontology
    }
    for s, _, t in g.triples((None, RDF.type, None)):
        if isinstance(s, URIRef) and str(s).startswith(BASE) and t not in non_individual_types:
            nodes.add(s)

    base_edges = []

    # TBox is-a edges.
    for child, _, parent in g.triples((None, RDFS.subClassOf, None)):
        if (isinstance(child, URIRef) and isinstance(parent, URIRef)
                and str(child).startswith(BASE) and str(parent).startswith(BASE)):
            nodes.update([child, parent])
            base_edges.append((child, "IS_A", parent))

    # OWL restrictions using eligible semantic properties.
    for child, _, r in g.triples((None, RDFS.subClassOf, None)):
        if not (isinstance(child, URIRef) and str(child).startswith(BASE) and isinstance(r, BNode)):
            continue
        if (r, RDF.type, OWL.Restriction) not in g:
            continue
        p = next(g.objects(r, OWL.onProperty), None)
        if p not in eligible_props:
            continue
        filler = None
        for pred in (OWL.someValuesFrom, OWL.allValuesFrom, OWL.hasValue):
            filler = next(g.objects(r, pred), None)
            if filler is not None:
                break
        if isinstance(filler, URIRef) and str(filler).startswith(BASE):
            nodes.update([child, filler])
            base_edges.append((child, local_name(p).upper(), filler))

    # Direct semantic assertions.
    for p in eligible_props:
        for s, _, o in g.triples((None, p, None)):
            if (isinstance(s, URIRef) and isinstance(o, URIRef)
                    and str(s).startswith(BASE) and str(o).startswith(BASE)):
                nodes.update([s, o])
                base_edges.append((s, local_name(p).upper(), o))

    base_edges = sorted(set(base_edges), key=lambda x: (str(x[0]), x[1], str(x[2])))

    # Add reverse relation types for message passing.
    edges = []
    for s, r, o in base_edges:
        edges.append((s, r, o))
        edges.append((o, r + "__REV", s))

    nodes = sorted(nodes, key=str)
    node_to_id = {n: i for i, n in enumerate(nodes)}
    rel_names = sorted({r for _, r, _ in edges})
    rel_to_id = {r: i for i, r in enumerate(rel_names)}

    src = torch.tensor([node_to_id[s] for s, _, _ in edges], dtype=torch.long)
    rel = torch.tensor([rel_to_id[r] for _, r, _ in edges], dtype=torch.long)
    dst = torch.tensor([node_to_id[o] for _, _, o in edges], dtype=torch.long)

    anchor_ids = []
    for c in CLASS_NAMES:
        u = SD[ANCHOR_CLASS_LOCAL[c]]
        if u not in node_to_id:
            raise KeyError(f"Missing recognition-state anchor: {u}")
        anchor_ids.append(node_to_id[u])

    return {
        "graph": g,
        "nodes": nodes,
        "node_to_id": node_to_id,
        "rel_names": rel_names,
        "src": src,
        "rel": rel,
        "dst": dst,
        "anchor_ids": torch.tensor(anchor_ids, dtype=torch.long),
        "specificity": torch.tensor(compute_specificity(g), dtype=torch.float32),
        "topology": torch.tensor(
            compute_anchor_topology(
                g, base_edges,
                tau_d=RECOGNITION_TOPOLOGY_TAU,
                allowed_relations=RECOGNITION_TOPOLOGY_RELATIONS,
                max_offdiag=RECOGNITION_TOPOLOGY_MAX_OFFDIAG,
            ), dtype=torch.float32
        ),
        "context_topology": torch.tensor(
            compute_full_context_topology(g, base_edges, tau_d=2.0), dtype=torch.float32
        ),
        "recognition_topology_relations": sorted(RECOGNITION_TOPOLOGY_RELATIONS),
    }


# ---------------------------------------------------------------------
# Relation-aware ontology encoder
# ---------------------------------------------------------------------
class RGCNLayer(nn.Module):
    def __init__(self, dim, num_relations, dropout=0.10):
        super().__init__()
        self.rel_weight = nn.Parameter(torch.empty(num_relations, dim, dim))
        self.self_weight = nn.Linear(dim, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.rel_weight)

    def forward(self, h, src, rel, dst):
        W = self.rel_weight[rel]                                # [E,D,D]
        msg = torch.bmm(h[src].unsqueeze(1), W).squeeze(1)     # [E,D]

        # Destination-relation degree normalization.
        key = dst * self.rel_weight.shape[0] + rel
        deg = torch.bincount(
            key, minlength=h.shape[0] * self.rel_weight.shape[0]
        ).float()
        msg = msg / deg[key].clamp_min(1.0).unsqueeze(1)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        out = self.self_weight(h) + agg + self.bias
        out = F.relu(self.norm(out))
        return self.dropout(out)


class RGCNOntologyEncoder(nn.Module):
    def __init__(self, num_nodes, num_relations, dim=256, layers=2, dropout=0.10):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, dim)
        nn.init.normal_(self.node_emb.weight, std=0.02)
        self.layers = nn.ModuleList([
            RGCNLayer(dim, num_relations, dropout) for _ in range(layers)
        ])

    def forward(self, src, rel, dst):
        h = self.node_emb.weight
        for layer in self.layers:
            h = h + layer(h, src, rel, dst)
        return F.normalize(h, dim=1)


# ---------------------------------------------------------------------
# Visual model
# ---------------------------------------------------------------------


class VisualModel(nn.Module):
    """
    OCRDA-v7 visual student.

    ConvNeXt-Tiny is retained from v6, but classification is explicitly
    decomposed into:
      - source_head: source-supervised decision geometry;
      - target_head: target-adapted decision geometry.

    The target head is synchronized from the source head immediately before
    adaptation begins, after the source-only warm-up.
    """
    def __init__(self, emb_dim=256, pretrained=True, dropout=0.20):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        net = convnext_tiny(weights=weights)
        in_dim = net.classifier[2].in_features
        net.classifier[2] = nn.Identity()
        self.backbone = net
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        self.source_head = nn.Linear(emb_dim, len(CLASS_NAMES))
        self.target_head = nn.Linear(emb_dim, len(CLASS_NAMES))
        self.sync_target_from_source()

    @torch.no_grad()
    def sync_target_from_source(self):
        self.target_head.load_state_dict(self.source_head.state_dict())

    def encode(self, x):
        feat = self.backbone(x)
        return F.normalize(self.proj(feat), dim=1)

    def forward(self, x, head="source", return_both=False):
        z = self.encode(x)
        source_logits = self.source_head(z)
        target_logits = self.target_head(z)
        if return_both:
            return z, source_logits, target_logits
        if head == "source":
            return z, source_logits
        if head == "target":
            return z, target_logits
        raise ValueError(f"Unknown classifier head: {head}")


@torch.no_grad()
def ema_update_module(teacher, student, momentum=0.996):
    """EMA teacher update; buffers are copied rather than exponentially mixed."""
    t_params = dict(teacher.named_parameters())
    s_params = dict(student.named_parameters())
    for name, tp in t_params.items():
        sp = s_params[name]
        tp.data.mul_(momentum).add_(sp.data, alpha=1.0 - momentum)
    t_buffers = dict(teacher.named_buffers())
    s_buffers = dict(student.named_buffers())
    for name, tb in t_buffers.items():
        if name in s_buffers:
            tb.copy_(s_buffers[name])


def curriculum_keep_ratio(epoch, warmup, epochs, start=0.25, end=0.50):
    """
    Conservative-to-broad target pseudo-label curriculum.
    The ratio is fixed at start during warm-up, then increases smoothly.
    """
    if epoch < warmup:
        return float(start)
    denom = max(epochs - warmup - 1, 1)
    p = min(max((epoch - warmup) / denom, 0.0), 1.0)
    # cosine increase
    a = 0.5 - 0.5 * math.cos(math.pi * p)
    return float(start + (end - start) * a)


def sharpen_distribution(q, temperature=0.70, eps=1e-8):
    if temperature <= 0:
        raise ValueError("pseudo-label temperature must be > 0")
    x = q.clamp_min(eps).pow(1.0 / temperature)
    return x / x.sum(dim=1, keepdim=True).clamp_min(eps)


def weighted_mean(loss_vec, weight, eps=1e-8):
    return (loss_vec * weight).sum() / weight.sum().clamp_min(eps)


def ontology_conditioned_prototypes(anchors, prototype_memory, memory_weight=0.60):
    """
    Build class prototypes by combining persistent target memory and trainable
    ontology anchors. This makes the contrastive target objective explicitly
    ontology-conditioned while retaining domain-specific target evidence.
    """
    protos = []
    for c in range(len(CLASS_NAMES)):
        a = F.normalize(anchors[c], dim=0)
        if bool(prototype_memory.initialized[c].item()):
            m = F.normalize(prototype_memory.values[c].detach(), dim=0)
            p = memory_weight * m + (1.0 - memory_weight) * a
        else:
            p = a
        protos.append(F.normalize(p, dim=0))
    return torch.stack(protos, dim=0)


def teacher_student_target_losses(
    z_student,
    logits_student,
    z_teacher,
    logits_teacher,
    student_anchors,
    teacher_anchors,
    prototype_memory,
    pseudo_thresholds,
    eta,
    tau_o,
    tau_js,
    pseudo_temperature=0.70,
    proto_temperature=0.10,
    proto_memory_weight=0.60,
):
    """
    Mean-Teacher target learning:
      1. teacher weak view produces ontology-fused pseudo-label distribution;
      2. class-specific global thresholds decide which pseudo-labels are used;
      3. student strong view is trained by weighted pseudo CE;
      4. a KL consistency loss preserves the full teacher distribution;
      5. ontology-conditioned prototype contrastive learning shapes the target
         representation geometry.

    No target labels are accessed.
    """
    with torch.no_grad():
        p_t = F.softmax(logits_teacher, dim=1)
        o_t = ontology_probs(z_teacher, teacher_anchors, tau_o)
        q_t = fused_distribution(p_t, o_t, eta=eta)
        q_t = sharpen_distribution(q_t, temperature=pseudo_temperature)
        r_t = reliability(p_t, o_t, q_t, tau_js)
        hard = q_t.argmax(dim=1)
        hard_score = r_t * q_t.gather(1, hard[:, None]).squeeze(1)
        class_thr = pseudo_thresholds[hard]
        selected = hard_score >= class_thr
        sel_w = r_t * selected.float()

    pseudo_ce_vec = F.cross_entropy(
        logits_student, hard, reduction="none"
    )
    l_pseudo = weighted_mean(pseudo_ce_vec, sel_w)

    # Distribution consistency also uses non-selected samples, but reliability
    # prevents uncertain target views from dominating.
    log_ps = F.log_softmax(logits_student, dim=1)
    kl_vec = F.kl_div(log_ps, q_t, reduction="none").sum(dim=1)
    l_teacher_consistency = weighted_mean(kl_vec, r_t.detach())

    proto = ontology_conditioned_prototypes(
        student_anchors,
        prototype_memory,
        memory_weight=proto_memory_weight,
    )
    proto_logits = (z_student @ proto.t()) / proto_temperature
    proto_ce_vec = F.cross_entropy(proto_logits, hard, reduction="none")
    l_proto = weighted_mean(proto_ce_vec, sel_w)

    stats = {
        "selected_pseudo_rate_batch": float(selected.float().mean().detach().cpu()),
        "mean_teacher_reliability_batch": float(r_t.mean().detach().cpu()),
        "mean_teacher_confidence_batch": float(
            q_t.max(dim=1).values.mean().detach().cpu()
        ),
        "pseudo_ce": float(l_pseudo.detach().cpu()),
        "teacher_consistency": float(l_teacher_consistency.detach().cpu()),
        "prototype_contrastive": float(l_proto.detach().cpu()),
    }
    return l_pseudo, l_teacher_consistency, l_proto, stats


# ---------------------------------------------------------------------
# OCRDA v7: prior-aware pseudo-labeling + ranking topology + normalized alignment
# ---------------------------------------------------------------------
def ontology_probs(z, anchors, tau):
    return F.softmax((z @ F.normalize(anchors, dim=1).t()) / tau, dim=1)


def fused_distribution(p, o, eta=0.90, eps=1e-8):
    """
    Geometric fusion of neural probabilities p and ontology compatibility o.
    Larger eta makes the visual classifier dominate when ontology anchors are
    not yet sufficiently calibrated.
    """
    q = p.clamp_min(eps).pow(eta) * o.clamp_min(eps).pow(1.0 - eta)
    return q / q.sum(dim=1, keepdim=True).clamp_min(eps)


def js_divergence(p, q, eps=1e-8):
    p, q = p.clamp_min(eps), q.clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * (
        (p * (p.log() - m.log())).sum(dim=1)
        + (q * (q.log() - m.log())).sum(dim=1)
    )


def reliability(p, o, q, tau_js=0.30, eps=1e-8):
    """
    Reliability combines:
      1) fused confidence,
      2) normalized entropy,
      3) neural-ontology Jensen-Shannon agreement.
    """
    maxq = q.max(dim=1).values
    H = -(q.clamp_min(eps) * q.clamp_min(eps).log()).sum(dim=1)
    entropy_term = (1.0 - H / math.log(q.shape[1])).clamp(0.0, 1.0)
    agreement = torch.exp(-js_divergence(p, o) / tau_js)
    return (maxq * entropy_term * agreement).clamp(0.0, 1.0)



def semantic_alignment_loss(zs, ys, anchors, tau=0.10):
    return F.cross_entropy((zs @ F.normalize(anchors, dim=1).t()) / tau, ys)


def ontology_agreement_loss(logits_t, zt, anchors, tau_o, tau_js, eta=0.90):
    p = F.softmax(logits_t, dim=1)
    o = ontology_probs(zt, anchors, tau_o)
    q = fused_distribution(p, o, eta=eta)
    r = reliability(p, o, q, tau_js).detach()
    return (r * js_divergence(p, o)).sum() / r.sum().clamp_min(1e-6)


def normalized_entropy(prob, eps=1e-8):
    H = -(prob.clamp_min(eps) * prob.clamp_min(eps).log()).sum(dim=-1)
    return H / math.log(prob.shape[-1])


def effective_class_count(mean_distribution, eps=1e-8):
    """
    exp(entropy) is 1 under complete class collapse and C under a uniform
    C-class distribution.
    """
    p = mean_distribution / mean_distribution.sum().clamp_min(eps)
    return float(torch.exp(-(p.clamp_min(eps) * p.clamp_min(eps).log()).sum()).item())


def dataset_class_prior(dataset, device=None, eps=1e-8):
    labels = [CLASS_TO_ID[normalize_label(dataset.df.iloc[i])] for i in range(len(dataset))]
    counts = torch.tensor(
        np.bincount(labels, minlength=len(CLASS_NAMES)), dtype=torch.float32
    )
    prior = counts / counts.sum().clamp_min(eps)
    return prior.to(device) if device is not None else prior


def temperature_smooth_prior(prior, temperature=1.50, eps=1e-8):
    """Flatten an uncertain class-prior estimate without forcing uniformity."""
    if temperature <= 0:
        raise ValueError("prior temperature must be > 0")
    x = prior.clamp_min(eps).pow(1.0 / temperature)
    return x / x.sum().clamp_min(eps)


def confidence_weighted_prior(prob, eps=1e-8):
    """Label-free class prior using entropy-derived confidence weights."""
    ent = normalized_entropy(prob, eps=eps)
    w = (1.0 - ent).clamp_min(0.05)
    p = (w[:, None] * prob).sum(dim=0) / w.sum().clamp_min(eps)
    return p / p.sum().clamp_min(eps)


def source_soft_confusion_from_eval(eval_pred, device, eps=1e-8):
    """Rows=true class, cols=mean source-teacher posterior on source validation."""
    if not eval_pred:
        return None
    y = torch.tensor(eval_pred.get("y_true", []), dtype=torch.long, device=device)
    probs = torch.tensor(eval_pred.get("probs", []), dtype=torch.float32, device=device)
    if y.numel() == 0 or probs.numel() == 0:
        return None
    rows=[]
    for c in range(len(CLASS_NAMES)):
        mask=(y==c)
        if int(mask.sum().item()) == 0:
            rows.append(torch.eye(len(CLASS_NAMES), device=device)[c])
        else:
            rows.append(probs[mask].mean(dim=0))
    M=torch.stack(rows,dim=0).clamp_min(eps)
    return M / M.sum(dim=1,keepdim=True).clamp_min(eps)


def bbse_prior_from_source_teacher(
    p_source_target, source_soft_confusion, source_prior, ridge=0.03, eps=1e-8
):
    """
    Soft BBSE-style target-prior estimate using only source validation labels and
    frozen-source predictions on unlabeled target data. Returns prior, fit error,
    condition number, and a conservative trust score.
    """
    if source_soft_confusion is None:
        return None, None, None, 0.0
    q = p_source_target.mean(dim=0)
    A = source_soft_confusion.t()  # predicted marginal = A @ true prior
    I = torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
    lhs = A.t() @ A + float(ridge) * I
    rhs = A.t() @ q + float(ridge) * source_prior
    try:
        p = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        p = torch.linalg.pinv(lhs) @ rhs
    p = p.clamp_min(0.0)
    if float(p.sum().item()) <= eps:
        p = source_prior.clone()
    p = p / p.sum().clamp_min(eps)
    fit = float(torch.abs(A @ p - q).sum().item())
    try:
        cond = float(torch.linalg.cond(A).item())
    except RuntimeError:
        cond = 1e6
    trust = math.exp(-fit / 0.12) * math.exp(-max(cond - 4.0, 0.0) / 12.0)
    trust = float(np.clip(trust, 0.0, 1.0))
    return p, fit, cond, trust


def robust_multievidence_prior(
    p_s, p_e, p_o, q, tri_score, eligible, consensus_class,
    source_soft_confusion, source_prior,
    w_consensus=0.35, w_teacher_median=0.25, w_fusion=0.25, w_bbse=0.15,
    eps=1e-8,
):
    """OCRDA-v7.4.4 robust label-free prior evidence ensemble."""
    # Reliability-weighted fused posterior.
    prior_w = tri_score.detach().clamp_min(0.0)
    fusion_prior = (prior_w[:, None] * q).sum(dim=0) / prior_w.sum().clamp_min(eps)
    fusion_prior = fusion_prior / fusion_prior.sum().clamp_min(eps)

    # Median of three independent confidence-weighted teachers resists one-head drift.
    teacher_priors = torch.stack([
        confidence_weighted_prior(p_s, eps),
        confidence_weighted_prior(p_e, eps),
        confidence_weighted_prior(p_o, eps),
    ], dim=0)
    teacher_median = teacher_priors.median(dim=0).values.clamp_min(eps)
    teacher_median = teacher_median / teacher_median.sum().clamp_min(eps)

    # High-confidence tri-consensus support distribution.
    if bool(eligible.any()):
        idx = torch.nonzero(eligible, as_tuple=False).flatten()
        cls = consensus_class[idx]
        ww = tri_score[idx].clamp_min(eps)
        cons = torch.zeros(len(CLASS_NAMES), dtype=q.dtype, device=q.device)
        cons.scatter_add_(0, cls, ww)
        consensus_prior = cons / cons.sum().clamp_min(eps)
        consensus_available = 1.0
    else:
        consensus_prior = teacher_median.clone()
        consensus_available = 0.0

    bbse, bbse_fit, bbse_cond, bbse_trust = bbse_prior_from_source_teacher(
        p_s, source_soft_confusion, source_prior
    )
    if bbse is None:
        bbse = teacher_median.clone()
        bbse_trust = 0.0

    weights = torch.tensor([
        float(w_consensus) * consensus_available,
        float(w_teacher_median),
        float(w_fusion),
        float(w_bbse) * float(bbse_trust),
    ], dtype=q.dtype, device=q.device)
    components = torch.stack([consensus_prior, teacher_median, fusion_prior, bbse], dim=0)
    if float(weights.sum().item()) <= eps:
        evidence = teacher_median
    else:
        evidence = (weights[:, None] * components).sum(dim=0) / weights.sum().clamp_min(eps)
    evidence = evidence.clamp_min(eps)
    evidence = evidence / evidence.sum().clamp_min(eps)
    diagnostics = {
        "prior_bbse_fit_l1": bbse_fit,
        "prior_bbse_condition": bbse_cond,
        "prior_bbse_trust": float(bbse_trust),
        "prior_consensus_available": float(consensus_available),
        "prior_component_weights": [float(x) for x in weights.detach().cpu().tolist()],
        "prior_teacher_median": teacher_median.detach(),
        "prior_consensus": consensus_prior.detach(),
        "prior_fusion": fusion_prior.detach(),
        "prior_bbse": bbse.detach(),
    }
    return evidence, diagnostics


def regularize_target_prior(
    observed_prior,
    prior_ema,
    source_prior,
    ema_momentum=0.70,
    source_weight=0.20,
    temperature=1.10,
    prior_floor=0.02,
    prior_ceiling=0.85,
    adaptive_tau=0.05,
    min_source_weight=0.005,
    eps=1e-8,
):
    """
    OCRDA-v7.4.4 label-shift-aware target-prior regularization.

    The incoming observed_prior is already a robust multi-evidence, label-free
    estimate combining tri-consensus, teacher median, fused posterior, and BBSE.

    Improvements over v7.4.1:
      1) the unlabeled target evidence is less aggressively flattened;
      2) the EMA follows target evidence faster;
      3) source-prior anchoring is automatically weakened when the current
         target estimate diverges from the source distribution.

    No target labels are used. The adaptive anchor is based only on the
    Jensen-Shannon divergence between the source prior and the unlabeled
    target estimate.
    """
    observed = temperature_smooth_prior(observed_prior, temperature, eps)
    updated_ema = ema_momentum * prior_ema + (1.0 - ema_momentum) * observed
    updated_ema = updated_ema / updated_ema.sum().clamp_min(eps)

    # Symmetric JS divergence in nats, bounded and stable.
    p = source_prior.clamp_min(eps)
    q = updated_ema.clamp_min(eps)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - m.log())).sum() + 0.5 * (q * (q.log() - m.log())).sum()

    # Strong source anchoring is useful only when target evidence is compatible
    # with the source prior. Under label shift it decays automatically.
    adaptive_weight = float(source_weight) * math.exp(-float(js.item()) / max(float(adaptive_tau), eps))
    adaptive_weight = float(np.clip(adaptive_weight, min_source_weight, source_weight))

    used = adaptive_weight * source_prior + (1.0 - adaptive_weight) * updated_ema
    used = used.clamp(min=prior_floor, max=prior_ceiling)
    used = used / used.sum().clamp_min(eps)
    return observed, updated_ema, used, adaptive_weight, float(js.item())



def decayed_source_prior_weight(
    epoch,
    epochs,
    start=0.60,
    end=0.15,
    schedule="cosine",
):
    """
    Label-free source-prior regularization schedule.

    v4 used a fixed source-prior weight. v5 gradually decreases this
    regularization so early pseudo-prior estimates are stabilized by the
    source distribution while later epochs are allowed to follow the
    unlabeled target evidence more strongly.

    cosine:
        w_s(e) = end + 0.5*(start-end)*(1 + cos(pi * e/(E-1)))

    linear:
        w_s(e) = start + (end-start)*e/(E-1)
    """
    if not (0.0 <= end <= 1.0 and 0.0 <= start <= 1.0):
        raise ValueError("source-prior weights must be in [0,1]")
    if start < end:
        raise ValueError("prior-source-weight-start should be >= end for decay")
    progress = 0.0 if epochs <= 1 else min(max(float(epoch) / float(epochs - 1), 0.0), 1.0)
    if schedule == "cosine":
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))
    if schedule == "linear":
        return start + (end - start) * progress
    if schedule == "exponential":
        # Smoothly approaches end while retaining the exact start at epoch 0.
        tau = max(float(epochs) / 3.0, 1.0)
        raw = math.exp(-float(epoch) / tau)
        raw_end = math.exp(-float(max(epochs - 1, 0)) / tau)
        scaled = (raw - raw_end) / max(1.0 - raw_end, 1e-8)
        return end + (start - end) * scaled
    raise ValueError(f"Unknown prior-source-weight schedule: {schedule}")


def prior_aware_thresholds(
    score,
    target_prior,
    total_keep_ratio=0.55,
    min_score=0.015,
    min_candidates=8,
    min_class_rate=0.02,
    max_class_rate=0.35,
):
    """
    Prior-Aware Adaptive Pseudo-labeling (PAPL).

    Let score[j,c] = r_j q_j(c). Instead of selecting a fixed 25% of target
    samples for every class, v4 allocates a TOTAL candidate budget according to
    a regularized unlabeled target-prior estimate.

        K_c ~= N * total_keep_ratio * pi_t(c)

    K_c is additionally bounded by minimum/maximum class rates. The classwise
    threshold is the K_c-th largest score, subject to a minimum reliability-
    semantic score floor. Multiple-class overlap is allowed but diagnosed.
    """
    if not (0.0 < total_keep_ratio <= 1.0):
        raise ValueError("total_keep_ratio must be in (0,1]")

    N, C = score.shape
    thresholds, masks, quotas = [], [], []

    for c in range(C):
        vals = score[:, c]
        desired = int(round(N * total_keep_ratio * float(target_prior[c].item())))
        lower = max(min_candidates, int(round(N * min_class_rate)))
        upper = max(lower, int(round(N * max_class_rate)))
        k = max(lower, min(desired, upper, N))
        quotas.append(k)

        if k <= 0 or vals.numel() == 0:
            thr = torch.tensor(float("inf"), device=score.device, dtype=score.dtype)
            mask = torch.zeros_like(vals, dtype=torch.bool)
        else:
            kth = torch.topk(vals, k=k, largest=True).values[-1]
            floor = torch.tensor(min_score, device=score.device, dtype=score.dtype)
            thr = torch.maximum(kth, floor)
            mask = vals >= thr

        thresholds.append(thr)
        masks.append(mask)

    return (
        torch.stack(thresholds),
        torch.stack(masks, dim=1),
        torch.tensor(quotas, device=score.device, dtype=torch.long),
    )


@torch.no_grad()
def target_pseudo_diagnostics(
    visual,
    rgcn,
    loader,
    src_e,
    rel_e,
    dst_e,
    anchor_ids,
    device,
    eta,
    tau_o,
    tau_js,
    prior_ema,
    source_prior,
    prior_ema_momentum,
    prior_source_weight,
    prior_temperature,
    prior_floor,
    prior_ceiling,
    total_keep_ratio,
    min_score,
    min_candidates,
    min_class_rate,
    max_class_rate,
    collapse_threshold,
):
    """Full unlabeled target-adaptation pass for PAPL and diagnostics."""
    visual.eval()
    rgcn.eval()

    H_ont = rgcn(src_e, rel_e, dst_e)
    anchors = H_ont[anchor_ids]

    Ps, Os, Qs, Rs, Zs, IDs = [], [], [], [], [], []
    for x, _, sid in loader:
        z, logits = visual(x.to(device))
        p = F.softmax(logits, dim=1)
        o = ontology_probs(z, anchors, tau_o)
        q = fused_distribution(p, o, eta=eta)
        r = reliability(p, o, q, tau_js)
        Ps.append(p); Os.append(o); Qs.append(q); Rs.append(r); Zs.append(z)
        IDs.extend(list(sid))

    p_all = torch.cat(Ps, dim=0)
    o_all = torch.cat(Os, dim=0)
    q_all = torch.cat(Qs, dim=0)
    r_all = torch.cat(Rs, dim=0)
    z_all = torch.cat(Zs, dim=0)
    score = r_all[:, None] * q_all

    q_mean = q_all.mean(dim=0)
    observed_prior, prior_ema_new, target_prior, prior_source_weight_effective, prior_source_js = regularize_target_prior(
        observed_prior=q_mean,
        prior_ema=prior_ema,
        source_prior=source_prior,
        ema_momentum=prior_ema_momentum,
        source_weight=prior_source_weight,
        temperature=prior_temperature,
        prior_floor=prior_floor,
        prior_ceiling=prior_ceiling,
    )

    thresholds, selected, quotas = prior_aware_thresholds(
        score=score,
        target_prior=target_prior,
        total_keep_ratio=total_keep_ratio,
        min_score=min_score,
        min_candidates=min_candidates,
        min_class_rate=min_class_rate,
        max_class_rate=max_class_rate,
    )

    hard = q_all.argmax(dim=1)
    hard_frac = torch.bincount(hard, minlength=len(CLASS_NAMES)).float()
    hard_frac = hard_frac / max(len(hard), 1)
    p_mean = p_all.mean(dim=0)
    o_mean = o_all.mean(dim=0)

    rel_by_class = []
    for c in range(len(CLASS_NAMES)):
        vals = r_all[hard == c]
        rel_by_class.append(float(vals.mean().item()) if vals.numel() else 0.0)

    selected_rate = selected.float().mean(dim=0)
    selected_mass = (score * selected.float()).sum(dim=0)
    selected_any = selected.any(dim=1).float().mean()
    selected_overlap = (selected.sum(dim=1) > 1).float().mean()

    eff = effective_class_count(q_mean)
    max_frac = float(hard_frac.max().item())
    collapse = bool(max_frac >= collapse_threshold or eff < 1.50)

    diagnostics = {
        "n_target_adapt": int(q_all.shape[0]),
        "mean_reliability": float(r_all.mean().item()),
        "mean_normalized_entropy_q": float(normalized_entropy(q_all).mean().item()),
        "effective_class_count_q": eff,
        "max_hard_pseudo_fraction": max_frac,
        "collapse_flag": collapse,
        "selected_any_rate": float(selected_any.item()),
        "selected_overlap_rate": float(selected_overlap.item()),
        "prior_source_weight_scheduled": float(prior_source_weight),
        "prior_source_weight_used": float(prior_source_weight_effective),
        "prior_source_js": float(prior_source_js),
    }

    for c, name in enumerate(CLASS_NAMES):
        diagnostics[f"p_mean_{name}"] = float(p_mean[c].item())
        diagnostics[f"o_mean_{name}"] = float(o_mean[c].item())
        diagnostics[f"q_mean_{name}"] = float(q_mean[c].item())
        diagnostics[f"hard_pseudo_fraction_{name}"] = float(hard_frac[c].item())
        diagnostics[f"reliability_hard_{name}"] = rel_by_class[c]
        diagnostics[f"source_prior_{name}"] = float(source_prior[c].item())
        diagnostics[f"observed_prior_{name}"] = float(observed_prior[c].item())
        diagnostics[f"prior_ema_{name}"] = float(prior_ema_new[c].item())
        diagnostics[f"target_prior_{name}"] = float(target_prior[c].item())
        for key, outname in [
            ("prior_teacher_median", "prior_teacher_median"),
            ("prior_consensus", "prior_consensus"),
            ("prior_fusion", "prior_fusion"),
            ("prior_bbse", "prior_bbse"),
        ]:
            vec = prior_component_diag.get(key)
            if vec is not None:
                diagnostics[f"{outname}_{name}"] = float(vec[c].item())
        diagnostics[f"candidate_quota_{name}"] = int(quotas[c].item())
        diagnostics[f"candidate_quota_rate_{name}"] = float(quotas[c].item() / max(len(q_all), 1))
        diagnostics[f"adaptive_threshold_{name}"] = float(thresholds[c].item())
        diagnostics[f"selected_rate_{name}"] = float(selected_rate[c].item())
        diagnostics[f"selected_fraction_of_selected_{name}"] = float(
            selected_dist[c].item()
        )
        diagnostics[f"selected_mass_{name}"] = float(selected_mass[c].item())

    snapshot = {
        "ids": IDs,
        "p": p_all.cpu(),
        "o": o_all.cpu(),
        "q": q_all.cpu(),
        "r": r_all.cpu(),
        "selected": selected.cpu(),
        "score": score.cpu(),
        "target_prior": target_prior.cpu(),
        "z": z_all.cpu(),
    }

    return (
        thresholds.detach(),
        target_prior.detach(),
        prior_ema_new.detach(),
        diagnostics,
        snapshot,
    )






class EMAPrototypeMemory:
    """
    Label-free target class-prototype memory.

    The persistent memory is detached from the autograd graph. During a
    training batch, a differentiable memory-assisted prototype is constructed
    as:

        mu_hat_c = (1-current_weight) * M_c.detach()
                   + current_weight * mu_batch_c

    whenever a usable current batch prototype exists. This preserves gradients
    through the current target features while making ontology pairwise
    constraints available even when a small mini-batch does not contain enough
    reliable pseudo-candidates for all classes.
    """
    def __init__(self, num_classes, emb_dim, device, momentum=0.90):
        self.num_classes = int(num_classes)
        self.emb_dim = int(emb_dim)
        self.device = device
        self.momentum = float(momentum)
        self.values = torch.zeros(self.num_classes, self.emb_dim, device=device)
        self.initialized = torch.zeros(self.num_classes, dtype=torch.bool, device=device)
        self.update_counts = torch.zeros(self.num_classes, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, class_id, prototype, momentum=None):
        if prototype is None:
            return
        c = int(class_id)
        proto = prototype.detach().to(self.device)
        m = self.momentum if momentum is None else float(momentum)
        if not bool(self.initialized[c].item()):
            self.values[c].copy_(proto)
            self.initialized[c] = True
        else:
            self.values[c].mul_(m).add_(proto, alpha=1.0 - m)
        self.update_counts[c] += 1

    @torch.no_grad()
    def update_many(self, prototypes, valid, momentum=None):
        for c, proto in enumerate(prototypes):
            if bool(valid[c]):
                self.update(c, proto, momentum=momentum)

    def memory_assisted(self, class_id, current_proto, current_valid, current_weight=0.35):
        c = int(class_id)
        has_mem = bool(self.initialized[c].item())
        mem = self.values[c].detach().clone() if has_mem else None
        if current_valid and current_proto is not None:
            if has_mem:
                w = float(current_weight)
                return (1.0 - w) * mem + w * current_proto, True, True
            return current_proto, True, True
        if has_mem:
            return mem, True, False
        return None, False, False

    def summary(self):
        return {
            "initialized_classes": int(self.initialized.sum().item()),
            "initialized_mask": [bool(x) for x in self.initialized.detach().cpu().tolist()],
            "update_counts": [int(x) for x in self.update_counts.detach().cpu().tolist()],
            "prototype_norms": [
                float(self.values[c].norm().detach().cpu()) if bool(self.initialized[c].item()) else 0.0
                for c in range(self.num_classes)
            ],
        }


@torch.no_grad()
def prototypes_from_snapshot(snapshot, min_mass=1.0):
    """
    Build full-target pseudo-prototypes from the unlabeled diagnostic pass.
    No target labels are used.

    Returns:
      prototypes: list[Tensor or None]
      valid: list[bool]
      masses: Tensor[C]
    """
    z = snapshot["z"]
    score = snapshot["score"]
    selected = snapshot["selected"].float()
    w_all = score * selected

    prototypes = []
    valid = []
    masses = []
    for c in range(len(CLASS_NAMES)):
        w = w_all[:, c]
        mass = w.sum()
        masses.append(mass)
        if float(mass.item()) >= float(min_mass):
            mu = (w[:, None] * z).sum(dim=0) / mass.clamp_min(1e-8)
            prototypes.append(mu)
            valid.append(True)
        else:
            prototypes.append(None)
            valid.append(False)
    return prototypes, valid, torch.stack(masses)


def ontology_ranking_topology_loss(
    mus_t,
    valid,
    ontology_similarity,
    rank_margin=0.10,
    rank_temperature=0.10,
    similarity_delta=0.05,
    separation_margin=0.30,
    separation_temperature=0.10,
):
    """
    Smooth ontology-ranking topology loss.

    Ontology structure does NOT prescribe an exact target cosine similarity.
    Instead, if ontology pair A is more similar than pair B, target feature
    centroids are encouraged to preserve the same relative ordering:

        d(A_close) + m_rank < d(A_far)

    A smooth softplus ranking penalty keeps a usable gradient even when a
    constraint is nearly satisfied. A separate smooth inter-class separation
    term prevents semantically related disease classes from collapsing.
    """
    pairs = []
    for i in range(len(CLASS_NAMES)):
        for j in range(i + 1, len(CLASS_NAMES)):
            if not (valid[i] and valid[j]):
                continue
            cos_sim = F.cosine_similarity(mus_t[i][None], mus_t[j][None]).squeeze()
            d = 1.0 - cos_sim
            s = ontology_similarity[i, j]
            pairs.append((i, j, s, d, cos_sim))

    if not pairs:
        return None, None, [], []

    rank_terms = []
    rank_info = []
    for a in range(len(pairs)):
        for b in range(len(pairs)):
            if a == b:
                continue
            i1, j1, s1, d1, _ = pairs[a]
            i2, j2, s2, d2, _ = pairs[b]
            if float((s1 - s2).detach().cpu()) <= similarity_delta:
                continue
            raw = d1 - d2 + rank_margin
            penalty = F.softplus(raw / rank_temperature) * rank_temperature
            rank_terms.append(penalty)
            rank_info.append({
                "closer_pair": f"{CLASS_NAMES[i1]}__{CLASS_NAMES[j1]}",
                "farther_pair": f"{CLASS_NAMES[i2]}__{CLASS_NAMES[j2]}",
                "ontology_similarity_close": float(s1.detach().cpu()),
                "ontology_similarity_far": float(s2.detach().cpu()),
                "feature_distance_close": float(d1.detach().cpu()),
                "feature_distance_far": float(d2.detach().cpu()),
                "penalty": float(penalty.detach().cpu()),
            })

    sep_terms = []
    pair_info = []
    for i, j, s, d, cos_sim in pairs:
        raw = separation_margin - d
        sep = F.softplus(raw / separation_temperature) * separation_temperature
        sep_terms.append(sep)
        pair_info.append({
            "pair": f"{CLASS_NAMES[i]}__{CLASS_NAMES[j]}",
            "cosine_similarity": float(cos_sim.detach().cpu()),
            "cosine_distance": float(d.detach().cpu()),
            "ontology_similarity": float(s.detach().cpu()),
            "separation_penalty": float(sep.detach().cpu()),
        })

    rank_loss = torch.stack(rank_terms).mean() if rank_terms else pairs[0][3] * 0.0
    sep_loss = torch.stack(sep_terms).mean() if sep_terms else pairs[0][3] * 0.0
    return rank_loss, sep_loss, rank_info, pair_info



@torch.no_grad()
def evaluate(model, loader, device, head="source"):
    model.eval()
    ys, ps, probs, ids = [], [], [], []
    for x, y, sid in loader:
        z, logits = model(x.to(device), head=head)
        prob = F.softmax(logits, dim=1).cpu().numpy()
        pred = prob.argmax(1)
        ys.extend(y.numpy().tolist())
        ps.extend(pred.tolist())
        probs.extend(prob.tolist())
        ids.extend(list(sid))

    precision, recall, f1, support = precision_recall_fscore_support(
        ys, ps, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(ys, ps)),
        "macro_f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(ys, ps)),
        "mcc": float(matthews_corrcoef(ys, ps)),
        "confusion_matrix": confusion_matrix(
            ys, ps, labels=list(range(len(CLASS_NAMES)))
        ).tolist(),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(CLASS_NAMES))
        },
    }
    return metrics, {"ids": ids, "y_true": ys, "y_pred": ps, "probs": probs}


@torch.no_grad()
def save_embeddings(model, loader, path, device, head="source"):
    model.eval()
    z_all, y_all, id_all = [], [], []
    for x, y, sid in loader:
        z, _ = model(x.to(device), head=head)
        z_all.append(z.cpu().numpy())
        y_all.extend(y.numpy().tolist())
        id_all.extend(list(sid))
    np.savez_compressed(
        path, z=np.concatenate(z_all), y=np.array(y_all), ids=np.array(id_all)
    )


def save_pseudo_snapshot(snapshot, path):
    data = {
        "sample_id": snapshot["ids"],
        "tri_consensus": snapshot["tri_consensus"].numpy().astype(int),
        "selected": snapshot["selected_any"].numpy().astype(int),
        "consensus_class": snapshot["consensus_class"].numpy(),
        "tri_score": snapshot["tri_score"].numpy(),
        "source_ema_js": snapshot["source_ema_js"].numpy(),
    }
    for c, name in enumerate(CLASS_NAMES):
        data[f"p_source_{name}"] = snapshot["p_source"][:, c].numpy()
        data[f"p_ema_{name}"] = snapshot["p_ema"][:, c].numpy()
        data[f"p_proto_{name}"] = snapshot["p_proto"][:, c].numpy()
        data[f"q_consensus_{name}"] = snapshot["q"][:, c].numpy()
        data[f"selected_{name}"] = snapshot["selected"][:, c].numpy().astype(int)
        data[f"target_prior_{name}"] = float(snapshot["target_prior"][c].item())
    pd.DataFrame(data).to_csv(path, index=False)



# ---------------------------------------------------------------------
# OCRDA-v7.4.4: source-anchored dual-teacher / ontology tri-consensus
# ---------------------------------------------------------------------

def v7_keep_ratio(epoch, warmup, epochs, start=0.10, end=0.35):
    """No target pseudo-labels during warm-up; conservative growth afterwards."""
    if epoch < warmup:
        return 0.0
    denom = max(epochs - warmup - 1, 1)
    p = min(max((epoch - warmup) / denom, 0.0), 1.0)
    a = 0.5 - 0.5 * math.cos(math.pi * p)
    return float(start + (end - start) * a)


def soft_semantic_target(class_id, ontology_similarity, epsilon=0.12, eps=1e-8):
    """
    Ontology-smoothed pseudo-label:
      (1-epsilon) on the consensus class and epsilon distributed to the other
      classes according to ontology similarity.
    """
    c = int(class_id)
    C = ontology_similarity.shape[0]
    y = torch.zeros(C, dtype=ontology_similarity.dtype, device=ontology_similarity.device)
    y[c] = 1.0 - epsilon
    off = ontology_similarity[c].clone()
    off[c] = 0.0
    if float(off.sum().detach().cpu()) <= eps:
        off = torch.ones_like(off)
        off[c] = 0.0
    off = off / off.sum().clamp_min(eps)
    y = y + epsilon * off
    return y / y.sum().clamp_min(eps)


def multi_geometric_fusion(probabilities, weights, eps=1e-8):
    if len(probabilities) != len(weights):
        raise ValueError("probabilities and weights must have same length")
    logq = 0.0
    total_w = float(sum(weights))
    for p, w in zip(probabilities, weights):
        logq = logq + (float(w) / total_w) * p.clamp_min(eps).log()
    q = torch.exp(logq)
    return q / q.sum(dim=1, keepdim=True).clamp_min(eps)


def source_anchored_prototype_bank(
    ontology_anchors,
    source_prototypes,
    prototype_memory,
    source_weight=0.55,
    memory_weight=0.25,
    ontology_weight=0.20,
):
    """
    P_c = normalize(w_s P_s^c + w_m M_t^c + w_o A_c^ont).

    Source prototypes are frozen. Target memory is detached. Ontology anchors
    may remain trainable in the student path, so ontology structure can still
    receive gradient through prototype/topology objectives.
    """
    bank = []
    for c in range(len(CLASS_NAMES)):
        a = F.normalize(ontology_anchors[c], dim=0)
        if source_prototypes is None:
            s = a.detach()
        else:
            s = F.normalize(source_prototypes[c].detach(), dim=0)
        parts = [source_weight * s, ontology_weight * a]
        total = source_weight + ontology_weight
        if prototype_memory is not None and bool(prototype_memory.initialized[c].item()):
            m = F.normalize(prototype_memory.values[c].detach(), dim=0)
            parts.append(memory_weight * m)
            total += memory_weight
        p = sum(parts) / max(total, 1e-8)
        bank.append(F.normalize(p, dim=0))
    return torch.stack(bank, dim=0)


@torch.no_grad()
def compute_source_prototypes(model, loader, device, head="source"):
    model.eval()
    sums = None
    counts = torch.zeros(len(CLASS_NAMES), device=device)
    for x, y, _ in loader:
        z, _ = model(x.to(device), head=head)
        y = y.to(device)
        if sums is None:
            sums = torch.zeros(len(CLASS_NAMES), z.shape[1], device=device)
        for c in range(len(CLASS_NAMES)):
            mask = y == c
            if mask.any():
                sums[c] += z[mask].sum(dim=0)
                counts[c] += mask.sum()
    if sums is None:
        raise RuntimeError("Source prototype loader is empty.")
    protos = sums / counts[:, None].clamp_min(1.0)
    return F.normalize(protos, dim=1), counts


@torch.no_grad()

def tempered_sampling_prior(target_prior, gamma=0.50, min_class_fraction=0.10, eps=1e-8):
    """
    OCRDA-v7.4.4: decouple label-shift calibration from pseudo-label sampling.

    `target_prior` remains the label-free estimate used for calibration and
    class-balanced losses.  Selection quotas instead use a tempered distribution
    p^gamma (gamma < 1), plus an explicit uniform floor, so a rare class cannot
    be starved merely because its estimated target prevalence is small.

    No target labels are used.  The final per-class quota is still capped by the
    number of candidates passing independent tri-consensus reliability gates.
    """
    p = target_prior.detach().float().clamp_min(eps)
    if not (0.0 < float(gamma) <= 1.0):
        raise ValueError("quota-temper-gamma must be in (0,1]")
    c = int(p.numel())
    floor = float(min_class_fraction)
    if floor < 0.0 or floor * c >= 1.0:
        raise ValueError("quota-min-class-fraction must satisfy 0 <= floor*C < 1")
    q = p.pow(float(gamma))
    q = q / q.sum().clamp_min(eps)
    if floor > 0:
        q = (1.0 - floor * c) * q + floor
    return q / q.sum().clamp_min(eps)


def _cpu_fp16_state_dict(module):
    """Compact CPU snapshot for rollback/ensemble candidates."""
    out = {}
    for k, v in module.state_dict().items():
        x = v.detach().cpu().clone()
        if torch.is_floating_point(x):
            x = x.half()
        out[k] = x
    return out


def _average_state_dicts(state_dicts):
    """SWA-style arithmetic mean of floating tensors; newest non-float buffer wins."""
    if not state_dicts:
        raise ValueError("No checkpoint states to average")
    keys = state_dicts[0].keys()
    avg = {}
    for k in keys:
        vals = [sd[k] for sd in state_dicts]
        if torch.is_floating_point(vals[0]):
            acc = vals[0].float().clone()
            for v in vals[1:]:
                acc.add_(v.float())
            acc.div_(len(vals))
            avg[k] = acc
        else:
            avg[k] = vals[-1].clone()
    return avg


def _hard_selected_collapse(diag, min_effective=2.0, max_fraction=0.75):
    eff = float(diag.get("selected_effective_class_count", 0.0))
    mx = float(diag.get("selected_max_class_fraction", 1.0))
    no_samples = int(diag.get("selected_total_count", 0)) <= 0
    bad = bool(no_samples or eff < float(min_effective) or mx > float(max_fraction))
    reasons=[]
    if no_samples: reasons.append("no_selected_samples")
    if eff < float(min_effective): reasons.append("selected_effective_classes_below_floor")
    if mx > float(max_fraction): reasons.append("selected_class_fraction_above_ceiling")
    return bad, reasons

def tri_consensus_diagnostics(
    source_teacher,
    ema_teacher,
    source_rgcn,
    loader,
    src_e,
    rel_e,
    dst_e,
    anchor_ids,
    device,
    source_prototypes,
    prototype_memory,
    ontology_similarity,
    source_prior,
    source_soft_confusion,
    prior_ema,
    prior_ema_momentum,
    prior_source_weight,
    prior_temperature,
    prior_floor,
    prior_ceiling,
    keep_ratio,
    source_proto_weight,
    memory_proto_weight,
    ontology_proto_weight,
    proto_vote_temperature,
    source_conf_threshold,
    ema_conf_threshold,
    proto_conf_threshold,
    source_ema_js_max,
    consensus_score_min,
    semantic_epsilon,
    fusion_source_weight,
    fusion_ema_weight,
    fusion_proto_weight,
    collapse_threshold,
    prior_evidence_weight_consensus=0.35,
    prior_evidence_weight_teacher_median=0.25,
    prior_evidence_weight_fusion=0.25,
    prior_evidence_weight_bbse=0.15,
    quota_temper_gamma=0.50,
    quota_min_class_fraction=0.10,
    quota_min_class_count=6,
):
    """
    Full unlabeled target pass.

    A target candidate is eligible only if three independent semantic votes
    agree:
      frozen source teacher == EMA target teacher == ontology prototype vote.

    Selection is then top-k within each consensus class under a global
    curriculum budget.  The budget uses a tempered sampling prior rather than
    the raw target prevalence estimate, with a label-free minimum class quota.
    Reliability gates are never relaxed to fill a quota.
    """
    source_teacher.eval()
    ema_teacher.eval()
    source_rgcn.eval()

    H_source = source_rgcn(src_e, rel_e, dst_e)
    source_anchors = H_source[anchor_ids]
    proto_bank = source_anchored_prototype_bank(
        source_anchors,
        source_prototypes,
        prototype_memory,
        source_weight=source_proto_weight,
        memory_weight=memory_proto_weight,
        ontology_weight=ontology_proto_weight,
    )

    Psrc, Pema, Pproto, Qs, Zs, IDs = [], [], [], [], [], []
    conf_s_all, conf_e_all, conf_o_all, js_all = [], [], [], []

    for x, _, sid in loader:
        x = x.to(device)
        z_s, log_s = source_teacher(x, head="source")
        z_e, log_e = ema_teacher(x, head="target")
        p_s = F.softmax(log_s, dim=1)
        p_e = F.softmax(log_e, dim=1)
        p_o = F.softmax((z_e @ proto_bank.t()) / proto_vote_temperature, dim=1)
        q = multi_geometric_fusion(
            [p_s, p_e, p_o],
            [fusion_source_weight, fusion_ema_weight, fusion_proto_weight],
        )
        Psrc.append(p_s); Pema.append(p_e); Pproto.append(p_o); Qs.append(q); Zs.append(z_e)
        conf_s_all.append(p_s.max(dim=1).values)
        conf_e_all.append(p_e.max(dim=1).values)
        conf_o_all.append(p_o.max(dim=1).values)
        js_all.append(js_divergence(p_s, p_e))
        IDs.extend(list(sid))

    p_s = torch.cat(Psrc, dim=0)
    p_e = torch.cat(Pema, dim=0)
    p_o = torch.cat(Pproto, dim=0)
    q = torch.cat(Qs, dim=0)
    z = torch.cat(Zs, dim=0)
    conf_s = torch.cat(conf_s_all)
    conf_e = torch.cat(conf_e_all)
    conf_o = torch.cat(conf_o_all)
    js_se = torch.cat(js_all)

    y_s = p_s.argmax(dim=1)
    y_e = p_e.argmax(dim=1)
    y_o = p_o.argmax(dim=1)
    tri = (y_s == y_e) & (y_e == y_o)
    consensus_class = y_e

    agree_score = torch.exp(-js_se / max(source_ema_js_max, 1e-6))
    tri_score = (
        conf_s.clamp_min(1e-8)
        * conf_e.clamp_min(1e-8)
        * conf_o.clamp_min(1e-8)
    ).pow(1.0 / 3.0) * agree_score

    eligible = (
        tri
        & (conf_s >= source_conf_threshold)
        & (conf_e >= ema_conf_threshold)
        & (conf_o >= proto_conf_threshold)
        & (js_se <= source_ema_js_max)
        & (tri_score >= consensus_score_min)
    )

    q_mean = q.mean(dim=0)

    # v7.4.4 robust label-shift evidence.  No target labels are used.
    prior_evidence, prior_component_diag = robust_multievidence_prior(
        p_s=p_s, p_e=p_e, p_o=p_o, q=q, tri_score=tri_score,
        eligible=eligible, consensus_class=consensus_class,
        source_soft_confusion=source_soft_confusion, source_prior=source_prior,
        w_consensus=prior_evidence_weight_consensus,
        w_teacher_median=prior_evidence_weight_teacher_median,
        w_fusion=prior_evidence_weight_fusion,
        w_bbse=prior_evidence_weight_bbse,
    )
    confidence_coverage = float((tri_score >= 0.35).float().mean().item())
    evidence_mix = 1.0

    observed_prior, prior_ema_new, target_prior, prior_source_weight_effective, prior_source_js = regularize_target_prior(
        observed_prior=prior_evidence,
        prior_ema=prior_ema,
        source_prior=source_prior,
        ema_momentum=prior_ema_momentum,
        source_weight=prior_source_weight,
        temperature=prior_temperature,
        prior_floor=prior_floor,
        prior_ceiling=prior_ceiling,
    )

    selected = torch.zeros(q.shape[0], len(CLASS_NAMES), dtype=torch.bool, device=device)
    quotas = torch.zeros(len(CLASS_NAMES), dtype=torch.long, device=device)
    desired_quotas = torch.zeros(len(CLASS_NAMES), dtype=torch.long, device=device)
    eligible_counts = torch.zeros(len(CLASS_NAMES), dtype=torch.long, device=device)
    thresholds = torch.full((len(CLASS_NAMES),), float("inf"), device=device)

    # v7.4.4: target prevalence is for calibration; pseudo-label sampling uses
    # a tempered distribution with a floor so a rare class is not starved.
    sampling_prior = tempered_sampling_prior(
        target_prior, gamma=quota_temper_gamma,
        min_class_fraction=quota_min_class_fraction,
    )
    if keep_ratio > 0:
        total_budget = int(round(q.shape[0] * keep_ratio))
        for c in range(len(CLASS_NAMES)):
            desired = int(round(total_budget * float(sampling_prior[c].item())))
            if total_budget >= len(CLASS_NAMES) * int(quota_min_class_count):
                desired = max(desired, int(quota_min_class_count))
            desired_quotas[c] = desired
            idx = torch.nonzero(eligible & (consensus_class == c), as_tuple=False).flatten()
            eligible_counts[c] = int(idx.numel())
            # Never manufacture low-reliability pseudo-labels merely to satisfy
            # class balance: quota is capped by independently eligible samples.
            k = min(desired, int(idx.numel()))
            quotas[c] = k
            if k > 0:
                vals = tri_score[idx]
                top = torch.topk(vals, k=k, largest=True)
                chosen = idx[top.indices]
                selected[chosen, c] = True
                thresholds[c] = top.values[-1]

    selected_any = selected.any(dim=1)
    selected_score = tri_score[selected_any]
    selected_rate = selected.float().mean(dim=0)

    # OCRDA-v7.4.4 checkpoint safety must describe the samples that actually
    # drive adaptation, not the hard argmax distribution over all unlabeled
    # target samples.  The v7 gate used q-hard statistics here and could reject
    # every checkpoint even when tri-consensus selection itself was healthy.
    selected_counts = selected.sum(dim=0).float()
    selected_total = selected_counts.sum()
    if float(selected_total.detach().cpu()) > 0:
        selected_dist = selected_counts / selected_total.clamp_min(1.0)
        selected_effective_classes = effective_class_count(selected_dist)
        selected_max_class_fraction = float(selected_dist.max().item())
    else:
        selected_dist = torch.zeros(
            len(CLASS_NAMES), dtype=torch.float32, device=device
        )
        selected_effective_classes = 0.0
        selected_max_class_fraction = 1.0

    hard = q.argmax(dim=1)
    hard_frac = torch.bincount(hard, minlength=len(CLASS_NAMES)).float()
    hard_frac = hard_frac / max(len(hard), 1)
    eff = effective_class_count(q_mean)
    max_frac = float(hard_frac.max().item())
    collapse = bool(max_frac >= collapse_threshold or eff < 1.50)

    diagnostics = {
        "n_target_adapt": int(q.shape[0]),
        "effective_class_count_q": eff,
        "max_hard_pseudo_fraction": max_frac,
        "collapse_flag": collapse,
        "tri_consensus_rate": float(tri.float().mean().item()),
        "tri_eligible_rate": float(eligible.float().mean().item()),
        "selected_any_rate": float(selected_any.float().mean().item()),
        "mean_source_ema_js": float(js_se.mean().item()),
        "mean_source_confidence": float(conf_s.mean().item()),
        "mean_ema_confidence": float(conf_e.mean().item()),
        "mean_proto_confidence": float(conf_o.mean().item()),
        "mean_tri_score": float(tri_score.mean().item()),
        "mean_selected_tri_score": (
            float(selected_score.mean().item()) if selected_score.numel() else 0.0
        ),
        "selected_effective_class_count": float(selected_effective_classes),
        "selected_max_class_fraction": float(selected_max_class_fraction),
        "selected_total_count": int(selected_total.item()),
        "prior_source_weight_scheduled": float(prior_source_weight),
        "prior_source_weight_used": float(prior_source_weight_effective),
        "prior_source_js": float(prior_source_js),
        "keep_ratio_used": float(keep_ratio),
        "prior_confidence_coverage": float(locals().get("confidence_coverage", 0.0)),
        "prior_evidence_mix": float(locals().get("evidence_mix", 0.0)),
        "prior_bbse_fit_l1": prior_component_diag.get("prior_bbse_fit_l1"),
        "prior_bbse_condition": prior_component_diag.get("prior_bbse_condition"),
        "prior_bbse_trust": prior_component_diag.get("prior_bbse_trust", 0.0),
        "prior_consensus_available": prior_component_diag.get("prior_consensus_available", 0.0),
        "quota_temper_gamma": float(quota_temper_gamma),
        "quota_min_class_fraction": float(quota_min_class_fraction),
        "quota_min_class_count": int(quota_min_class_count),
    }

    for c, name in enumerate(CLASS_NAMES):
        diagnostics[f"q_mean_{name}"] = float(q_mean[c].item())
        diagnostics[f"hard_pseudo_fraction_{name}"] = float(hard_frac[c].item())
        diagnostics[f"source_prior_{name}"] = float(source_prior[c].item())
        diagnostics[f"observed_prior_{name}"] = float(observed_prior[c].item())
        diagnostics[f"prior_ema_{name}"] = float(prior_ema_new[c].item())
        diagnostics[f"target_prior_{name}"] = float(target_prior[c].item())
        diagnostics[f"sampling_prior_{name}"] = float(sampling_prior[c].item())
        diagnostics[f"desired_quota_{name}"] = int(desired_quotas[c].item())
        diagnostics[f"eligible_count_{name}"] = int(eligible_counts[c].item())
        diagnostics[f"candidate_quota_{name}"] = int(quotas[c].item())
        diagnostics[f"candidate_quota_rate_{name}"] = float(quotas[c].item() / max(len(q), 1))
        diagnostics[f"quota_fill_ratio_{name}"] = float(quotas[c].item() / max(int(desired_quotas[c].item()), 1))
        diagnostics[f"consensus_threshold_{name}"] = float(thresholds[c].item())
        diagnostics[f"selected_rate_{name}"] = float(selected_rate[c].item())

    snapshot = {
        "ids": IDs,
        "p_source": p_s.cpu(),
        "p_ema": p_e.cpu(),
        "p_proto": p_o.cpu(),
        "q": q.cpu(),
        "z": z.cpu(),
        "tri_consensus": tri.cpu(),
        "eligible": eligible.cpu(),
        "consensus_class": consensus_class.cpu(),
        "tri_score": tri_score.cpu(),
        "source_ema_js": js_se.cpu(),
        "selected": selected.cpu(),
        "selected_any": selected_any.cpu(),
        "target_prior": target_prior.cpu(),
        "sampling_prior": sampling_prior.cpu(),
    }

    return target_prior.detach(), prior_ema_new.detach(), diagnostics, snapshot


@torch.no_grad()
def consensus_prototypes_from_snapshot(snapshot, min_selected=1):
    z = snapshot["z"]
    selected = snapshot["selected"]
    tri_score = snapshot["tri_score"]
    protos, valid, masses = [], [], []
    for c in range(len(CLASS_NAMES)):
        mask = selected[:, c]
        if int(mask.sum()) < int(min_selected):
            protos.append(torch.zeros(z.shape[1], dtype=z.dtype))
            valid.append(False)
            masses.append(torch.tensor(0.0))
            continue
        w = tri_score * mask.float()
        mass = w.sum()
        proto = (w[:, None] * z).sum(dim=0) / mass.clamp_min(1e-8)
        protos.append(proto)
        valid.append(True)
        masses.append(mass)
    return protos, valid, torch.stack(masses)


def build_consensus_cache(snapshot, ontology_similarity, semantic_epsilon=0.12):
    cache = {}
    for i, sid in enumerate(snapshot["ids"]):
        if not bool(snapshot["selected_any"][i].item()):
            continue
        c = int(snapshot["consensus_class"][i].item())
        cache[str(sid)] = {
            "class_id": c,
            "weight": float(snapshot["tri_score"][i].item()),
            "soft_target": soft_semantic_target(
                c, ontology_similarity, epsilon=semantic_epsilon
            ).detach().cpu(),
        }
    return cache


def batch_cache_tensors(sample_ids, cache, topology, device):
    B = len(sample_ids)
    hard = torch.full((B,), -1, dtype=torch.long, device=device)
    weight = torch.zeros(B, dtype=torch.float32, device=device)
    soft = torch.zeros(B, len(CLASS_NAMES), dtype=torch.float32, device=device)
    for i, sid in enumerate(sample_ids):
        rec = cache.get(str(sid))
        if rec is None:
            continue
        hard[i] = int(rec["class_id"])
        weight[i] = float(rec["weight"])
        soft[i] = rec["soft_target"].to(device)
    selected = hard >= 0
    return selected, hard, weight, soft



def class_balance_weights_from_prior(
    target_prior,
    gamma=0.75,
    min_weight=0.60,
    max_weight=1.80,
):
    """
    Label-free class weights from the regularized target prior.

    Low-prior classes receive more semantic adaptation gradient.
    Weights are normalized to mean 1 and clipped for stability.
    """
    p = target_prior.detach().float().clamp_min(1e-6)

    w = p.pow(-float(gamma))
    w = w / w.mean().clamp_min(1e-6)

    return w.clamp(
        min=float(min_weight),
        max=float(max_weight),
    )


def soft_cross_entropy(logits, soft_target):
    return -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1)



def consensus_target_losses(
    logits_target_student,
    z_target_student,
    logits_target_teacher,
    logits_source_teacher,
    sample_ids,
    pseudo_cache,
    student_anchors,
    source_prototypes,
    prototype_memory,
    ontology_similarity,
    source_proto_weight,
    memory_proto_weight,
    ontology_proto_weight,
    proto_temperature,
    target_prior,
    class_balance_gamma=0.75,
    class_balance_min=0.60,
    class_balance_max=1.80,
):
    selected, hard, weight, soft = batch_cache_tensors(
        sample_ids,
        pseudo_cache,
        ontology_similarity,
        logits_target_student.device,
    )

    zero = logits_target_student.sum() * 0.0

    if not selected.any():
        return zero, zero, zero, zero, {
            "selected_pseudo_rate_batch": 0.0,
            "mean_selected_weight_batch": 0.0,
            "mean_semantic_balanced_weight_batch": 0.0,
            "soft_pseudo_ce": 0.0,
            "ema_consistency": 0.0,
            "source_teacher_consistency": 0.0,
            "prototype_contrastive": 0.0,
        }, selected, hard, weight

    base_w = weight[selected].clamp_min(1e-6)

    class_w = class_balance_weights_from_prior(
        target_prior,
        gamma=class_balance_gamma,
        min_weight=class_balance_min,
        max_weight=class_balance_max,
    ).to(logits_target_student.device)

    semantic_w = (
        base_w
        * class_w[hard[selected]]
    ).clamp_min(1e-6)

    # Class-balanced soft semantic pseudo CE
    l_soft_vec = soft_cross_entropy(
        logits_target_student[selected],
        soft[selected],
    )
    l_soft = weighted_mean(
        l_soft_vec,
        semantic_w,
    )

    # Teacher consistency remains reliability-weighted rather
    # than strongly class-reweighted.
    with torch.no_grad():
        p_ema = F.softmax(
            logits_target_teacher[selected],
            dim=1,
        )
        p_source = F.softmax(
            logits_source_teacher[selected],
            dim=1,
        )

    log_student = F.log_softmax(
        logits_target_student[selected],
        dim=1,
    )

    l_ema_vec = F.kl_div(
        log_student,
        p_ema,
        reduction="none",
    ).sum(dim=1)

    l_ema = weighted_mean(
        l_ema_vec,
        base_w,
    )

    l_source_vec = F.kl_div(
        log_student,
        p_source,
        reduction="none",
    ).sum(dim=1)

    l_source = weighted_mean(
        l_source_vec,
        base_w,
    )

    proto_bank = source_anchored_prototype_bank(
        student_anchors,
        source_prototypes,
        prototype_memory,
        source_weight=source_proto_weight,
        memory_weight=memory_proto_weight,
        ontology_weight=ontology_proto_weight,
    )

    proto_logits = (
        z_target_student[selected]
        @ proto_bank.t()
    ) / proto_temperature

    proto_vec = F.cross_entropy(
        proto_logits,
        hard[selected],
        reduction="none",
    )

    # Semantic prototype learning is also class balanced.
    l_proto = weighted_mean(
        proto_vec,
        semantic_w,
    )

    # Topology receives the balanced target weights too.
    balanced_batch_weight = weight.clone()
    balanced_batch_weight[selected] = semantic_w

    stats = {
        "selected_pseudo_rate_batch":
            float(selected.float().mean().detach().cpu()),

        "mean_selected_weight_batch":
            float(base_w.mean().detach().cpu()),

        "mean_semantic_balanced_weight_batch":
            float(semantic_w.mean().detach().cpu()),

        "soft_pseudo_ce":
            float(l_soft.detach().cpu()),

        "ema_consistency":
            float(l_ema.detach().cpu()),

        "source_teacher_consistency":
            float(l_source.detach().cpu()),

        "prototype_contrastive":
            float(l_proto.detach().cpu()),
    }

    return (
        l_soft,
        l_ema,
        l_source,
        l_proto,
        stats,
        selected,
        hard,
        balanced_batch_weight,
    )


def v7_mean_topology_loss(
    zs,
    ys,
    zt,
    selected,
    hard,
    sample_weight,
    student_anchors,
    source_prototypes,
    prototype_memory,
    ontology_similarity,
    target_prior,
    source_proto_weight,
    memory_proto_weight,
    ontology_proto_weight,
    current_target_weight=0.20,
    rank_margin=0.10,
    rank_temperature=0.10,
    topology_similarity_delta=0.05,
):
    """
    V7 removes covariance matching entirely.

    The alignment branch contains only class-conditional cosine mean alignment,
    while ontology topology is enforced on source-anchored prototypes that may
    include current target evidence and therefore propagate gradient.
    """
    device = zs.device
    mean_align = torch.tensor(0.0, device=device)
    align_w = torch.tensor(0.0, device=device)

    protos = []
    valid = []
    grad_flags = []

    for c in range(len(CLASS_NAMES)):
        src_mask = ys == c
        mu_s = zs[src_mask].mean(dim=0) if src_mask.any() else None

        tgt_mask = selected & (hard == c)
        mu_t = None
        if tgt_mask.any():
            ww = sample_weight[tgt_mask].detach().clamp_min(1e-6)
            mu_t = (ww[:, None] * zt[tgt_mask]).sum(dim=0) / ww.sum()

        if mu_s is not None and mu_t is not None:
            w_c = target_prior[c].detach().clamp_min(1e-4)
            mean_align = mean_align + w_c * (
                1.0 - F.cosine_similarity(mu_s[None], mu_t[None]).squeeze()
            )
            align_w = align_w + w_c

        base_bank = source_anchored_prototype_bank(
            student_anchors,
            source_prototypes,
            prototype_memory,
            source_weight=source_proto_weight,
            memory_weight=memory_proto_weight,
            ontology_weight=ontology_proto_weight,
        )
        p = base_bank[c]
        has_current = mu_t is not None
        if has_current:
            p = F.normalize(
                (1.0 - current_target_weight) * p
                + current_target_weight * F.normalize(mu_t, dim=0),
                dim=0,
            )
        protos.append(p)
        valid.append(True)
        grad_flags.append(bool(has_current))

    if float(align_w.detach().cpu()) > 0:
        mean_align = mean_align / align_w

    rank_loss, _, _, pair_info = ontology_ranking_topology_loss(
        mus_t=protos,
        valid=valid,
        ontology_similarity=ontology_similarity,
        rank_margin=rank_margin,
        rank_temperature=rank_temperature,
        similarity_delta=topology_similarity_delta,
        separation_margin=0.0,
        separation_temperature=0.10,
    )
    if rank_loss is None:
        rank_loss = torch.tensor(0.0, device=device)

    gradient_pairs = 0
    for item in pair_info:
        names = item["pair"].split("__")
        ids = [CLASS_TO_ID[n] for n in names]
        if any(grad_flags[i] for i in ids):
            gradient_pairs += 1

    return mean_align, rank_loss, {
        "mean_alignment": float(mean_align.detach().cpu()),
        "ranking_topology": float(rank_loss.detach().cpu()),
        "topology_gradient_classes_batch": int(sum(grad_flags)),
        "topology_pairs_batch": int(len(pair_info)),
        "topology_pairs_with_current_gradient_batch": int(gradient_pairs),
        "prototype_memory_initialized_classes": int(
            prototype_memory.initialized.sum().item()
        ),
    }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def ramp(epoch, warmup, epochs):
    if epoch < warmup:
        return 0.0
    return min(1.0, (epoch - warmup + 1) / max(epochs - warmup, 1))


def _gate_high_good(x, low, high):
    """Continuous [0,1] gate: larger x is better."""
    if high <= low:
        return 1.0 if x >= high else 0.0
    return float(np.clip((x - low) / (high - low), 0.0, 1.0))


def _gate_low_good(x, good, bad):
    """Continuous [0,1] gate: smaller x is better."""
    if bad <= good:
        return 1.0 if x <= good else 0.0
    return float(np.clip((bad - x) / (bad - good), 0.0, 1.0))



# ---------------------------------------------------------------------
# OCRDA-v7.4.4: actual rolling plateau / drift controller
# ---------------------------------------------------------------------
def _v742_selected_distribution(diag):
    """Normalized distribution of samples that actually drive adaptation."""
    x = np.asarray([
        max(0.0, float(diag.get(f"selected_rate_{c}", 0.0)))
        for c in CLASS_NAMES
    ], dtype=np.float64)
    s = float(x.sum())
    if s <= 0.0:
        return np.zeros(len(CLASS_NAMES), dtype=np.float64)
    return x / s


def _v742_tv_distance(p, q):
    """Total-variation distance in [0,1]."""
    if p is None or q is None:
        return None
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.size == 0 or q.size == 0 or p.sum() <= 0 or q.sum() <= 0:
        return None
    return float(0.5 * np.abs(p - q).sum())


def _v742_mean_cosine_drift(cur, prev, cur_mask=None, prev_mask=None):
    """Mean 1-cosine drift for rows available in both tensors."""
    if cur is None or prev is None:
        return None
    cur = cur.detach().float().cpu()
    prev = prev.detach().float().cpu()
    if cur.shape != prev.shape or cur.ndim != 2:
        return None
    mask = torch.ones(cur.shape[0], dtype=torch.bool)
    if cur_mask is not None:
        mask &= cur_mask.detach().bool().cpu()
    if prev_mask is not None:
        mask &= prev_mask.detach().bool().cpu()
    if int(mask.sum().item()) == 0:
        return None
    a = F.normalize(cur[mask], dim=1)
    b = F.normalize(prev[mask], dim=1)
    drift = 1.0 - (a * b).sum(dim=1)
    return float(drift.clamp_min(0.0).mean().item())


@torch.no_grad()
def _v742_source_anchor_drift(
    teacher_rgcn, source_teacher_rgcn,
    src_e, rel_e, dst_e, anchor_ids,
):
    """Absolute semantic-anchor drift from the frozen source teacher."""
    if teacher_rgcn is None or source_teacher_rgcn is None:
        return None
    teacher_rgcn.eval()
    source_teacher_rgcn.eval()
    cur = teacher_rgcn(src_e, rel_e, dst_e)[anchor_ids]
    ref = source_teacher_rgcn(src_e, rel_e, dst_e)[anchor_ids]
    cur = F.normalize(cur, dim=1)
    ref = F.normalize(ref, dim=1)
    return float((1.0 - (cur * ref).sum(dim=1)).clamp_min(0.0).mean().item())


def _v742_geometric(values, weights, eps=1e-6):
    wsum = max(float(sum(weights)), eps)
    return float(np.clip(math.exp(sum(
        float(w) * math.log(max(float(v), eps))
        for v, w in zip(values, weights)
    ) / wsum), 0.0, 1.0))


def v742_update_controller(
    state,
    post_diag,
    prototype_memory,
    teacher_rgcn,
    source_teacher_rgcn,
    src_e, rel_e, dst_e, anchor_ids,
    source_val_macro_f1,
    source_retention_reference,
    args,
    adaptation_active,
):
    """
    Compute *actual* label-free drift/stability signals after an epoch.

    The returned multiplier is used on the NEXT epoch, so the controller is
    causal and cannot use post-epoch information to change the epoch that just
    finished. Target-development labels are never consulted.
    """
    tri_score = float(post_diag.get("mean_selected_tri_score", 0.0))
    source_ema_js = max(0.0, float(post_diag.get("mean_source_ema_js", 1.0)))
    selected_rate = float(post_diag.get("selected_any_rate", 0.0))

    selected_dist = _v742_selected_distribution(post_diag)
    selected_distribution_drift = _v742_tv_distance(
        selected_dist, state.get("prev_selected_distribution")
    )

    cur_proto = prototype_memory.values.detach().cpu().clone()
    cur_init = prototype_memory.initialized.detach().cpu().clone()
    prototype_drift = _v742_mean_cosine_drift(
        cur_proto,
        state.get("prev_prototype_values"),
        cur_init,
        state.get("prev_prototype_initialized"),
    )

    source_anchor_drift = _v742_source_anchor_drift(
        teacher_rgcn, source_teacher_rgcn,
        src_e, rel_e, dst_e, anchor_ids,
    )

    # Missing first-observation drifts are deliberately treated as not mature.
    sel_stability = (
        _gate_low_good(
            selected_distribution_drift,
            args.v742_selected_drift_good,
            args.v742_selected_drift_bad,
        ) if selected_distribution_drift is not None else 0.0
    )
    proto_stability = (
        _gate_low_good(
            prototype_drift,
            args.v742_prototype_drift_good,
            args.v742_prototype_drift_bad,
        ) if prototype_drift is not None else 0.0
    )
    anchor_stability = (
        _gate_low_good(
            source_anchor_drift,
            args.v742_source_anchor_drift_good,
            args.v742_source_anchor_drift_bad,
        ) if source_anchor_drift is not None else 0.0
    )
    tri_quality = _gate_high_good(
        tri_score, args.v742_tri_score_low, args.v742_tri_score_high
    )
    js_stability = _gate_low_good(
        source_ema_js, args.v742_js_good, args.v742_js_bad
    )

    maturity = _v742_geometric(
        [tri_quality, js_stability, sel_stability, proto_stability, anchor_stability],
        [args.v742_maturity_weight_tri,
         args.v742_maturity_weight_js,
         args.v742_maturity_weight_selected_stability,
         args.v742_maturity_weight_prototype_stability,
         args.v742_maturity_weight_anchor_stability],
        eps=args.adapt_gate_epsilon,
    )

    if adaptation_active and selected_rate > 0.0:
        state["post_adaptation_observations"] = int(
            state.get("post_adaptation_observations", 0)
        ) + 1

    history_ready = (
        int(state.get("post_adaptation_observations", 0))
        >= args.v742_min_history_epochs
    )

    mature_now = (
        history_ready
        and selected_rate >= args.checkpoint_min_selected_rate
        and maturity >= args.v742_maturity_threshold
    )

    if mature_now:
        state["plateau_counter"] = int(state.get("plateau_counter", 0)) + 1
    else:
        # Hysteresis: one noisy epoch does not erase established evidence.
        state["plateau_counter"] = max(
            0, int(state.get("plateau_counter", 0)) - 1
        )

    if (
        not state.get("plateau_active", False)
        and state["plateau_counter"] >= args.v742_plateau_patience
    ):
        state["plateau_active"] = True
        state["plateau_age"] = 0

    if state.get("plateau_active", False):
        state["plateau_age"] = int(state.get("plateau_age", 0)) + 1
        plateau_factor = max(
            args.v742_plateau_floor,
            args.v742_plateau_decay ** state["plateau_age"],
        )
    else:
        plateau_factor = 1.0

    source_retention_drop = 0.0
    if source_retention_reference is not None:
        source_retention_drop = max(
            0.0,
            float(source_retention_reference) - float(source_val_macro_f1),
        )
    source_retention_raw = _gate_low_good(
        source_retention_drop,
        args.v742_source_retention_drop_good,
        args.v742_source_retention_drop_bad,
    )
    source_retention_trust = max(
        args.v742_source_retention_floor, source_retention_raw
    )

    # Drift shock guard. Stable runs stay at 1.0; abrupt changes attenuate the
    # next epoch without permanently disabling adaptation.
    drift_stabilities = [js_stability, sel_stability, proto_stability, anchor_stability]
    min_stability = min(drift_stabilities) if history_ready else 1.0
    if history_ready and min_stability < args.v742_drift_alarm_stability:
        frac = min_stability / max(args.v742_drift_alarm_stability, 1e-8)
        drift_guard = max(
            args.v742_drift_guard_floor,
            args.v742_drift_guard_floor
            + (1.0 - args.v742_drift_guard_floor) * frac,
        )
    else:
        drift_guard = 1.0

    next_multiplier = float(np.clip(
        plateau_factor * source_retention_trust * drift_guard,
        args.v742_controller_floor,
        1.0,
    ))

    state["next_multiplier"] = next_multiplier
    state["prev_selected_distribution"] = selected_dist.copy()
    state["prev_prototype_values"] = cur_proto.clone()
    state["prev_prototype_initialized"] = cur_init.clone()

    return {
        "v742_selected_tri_score": tri_score,
        "v742_source_ema_js": source_ema_js,
        "v742_selected_distribution_drift": (
            float(selected_distribution_drift)
            if selected_distribution_drift is not None else np.nan
        ),
        "v742_prototype_drift": (
            float(prototype_drift) if prototype_drift is not None else np.nan
        ),
        "v742_source_anchor_drift": (
            float(source_anchor_drift) if source_anchor_drift is not None else np.nan
        ),
        "v742_source_retention_drop": float(source_retention_drop),
        "v742_tri_quality": float(tri_quality),
        "v742_js_stability": float(js_stability),
        "v742_selected_distribution_stability": float(sel_stability),
        "v742_prototype_stability": float(proto_stability),
        "v742_source_anchor_stability": float(anchor_stability),
        "v742_adaptation_maturity": float(maturity),
        "v742_history_ready": bool(history_ready),
        "v742_plateau_counter": int(state.get("plateau_counter", 0)),
        "v742_plateau_age": int(state.get("plateau_age", 0)),
        "v742_plateau_factor": float(plateau_factor),
        "v742_plateau_active": bool(state.get("plateau_active", False)),
        "v742_source_retention_trust": float(source_retention_trust),
        "v742_drift_guard": float(drift_guard),
        "v742_next_controller_multiplier": float(next_multiplier),
    }


def selected_distribution_adaptation_gate(diag, args):
    """
    OCRDA-v7.4.4 dual-distribution continuous adaptation gate.

    Selected pseudo-label quality remains the dominant signal, but the
    complete unlabeled target distribution also contributes a soft
    anti-collapse penalty.

        gate =
            selected_gate ** alpha
            * global_gate ** (1-alpha)

    No target labels are used.
    Global collapse never hard-disables adaptation.
    """

    selected_rate = float(diag.get("selected_any_rate", 0.0))
    tri_rate = float(diag.get("tri_consensus_rate", 0.0))

    selected_eff = float(
        diag.get("selected_effective_class_count", 0.0)
    )
    selected_max = float(
        diag.get("selected_max_class_fraction", 1.0)
    )

    tri_score = float(
        diag.get("mean_selected_tri_score", 0.0)
    )
    source_ema_js = max(
        0.0,
        float(diag.get("mean_source_ema_js", 1.0))
    )

    global_eff = float(
        diag.get("effective_class_count_q", 0.0)
    )
    global_max = float(
        diag.get("max_hard_pseudo_fraction", 1.0)
    )

    ready = (
        selected_rate >= args.adapt_gate_ready_selected_rate
        and tri_rate >= args.adapt_gate_ready_consensus_rate
        and int(diag.get("selected_total_count", 0)) > 0
    )

    # ----- selected-distribution gate -----
    g_eff = _gate_high_good(
        selected_eff,
        args.adapt_gate_eff_low,
        args.adapt_gate_eff_high,
    )

    g_bal = _gate_low_good(
        selected_max,
        args.adapt_gate_maxfrac_good,
        args.adapt_gate_maxfrac_bad,
    )

    g_tri = _gate_high_good(
        tri_score,
        args.adapt_gate_triscore_low,
        args.adapt_gate_triscore_high,
    )

    g_js = _gate_low_good(
        source_ema_js,
        args.adapt_gate_js_good,
        args.adapt_gate_js_bad,
    )

    g_sel = _gate_high_good(
        selected_rate,
        args.adapt_gate_selected_rate_low,
        args.adapt_gate_selected_rate_high,
    )

    selected_vals = [g_eff, g_bal, g_tri, g_js, g_sel]

    selected_weights = [
        args.adapt_gate_weight_eff,
        args.adapt_gate_weight_balance,
        args.adapt_gate_weight_tri,
        args.adapt_gate_weight_js,
        args.adapt_gate_weight_coverage,
    ]

    sw = max(sum(selected_weights), 1e-8)

    selected_log = sum(
        w * math.log(max(v, args.adapt_gate_epsilon))
        for v, w in zip(selected_vals, selected_weights)
    ) / sw

    selected_raw = float(
        np.clip(math.exp(selected_log), 0.0, 1.0)
    )

    # ----- global distribution safety -----
    g_global_eff = _gate_high_good(
        global_eff,
        args.dual_gate_global_eff_low,
        args.dual_gate_global_eff_high,
    )

    g_global_bal = _gate_low_good(
        global_max,
        args.dual_gate_global_maxfrac_good,
        args.dual_gate_global_maxfrac_bad,
    )

    global_raw = math.sqrt(
        max(g_global_eff, args.adapt_gate_epsilon)
        * max(g_global_bal, args.adapt_gate_epsilon)
    )

    alpha = float(
        np.clip(args.dual_gate_selected_mix, 0.0, 1.0)
    )

    combined_raw = (
        max(selected_raw, args.adapt_gate_epsilon) ** alpha
        * max(global_raw, args.adapt_gate_epsilon)
          ** (1.0 - alpha)
    )

    components = {
        "gate_selected_effective_classes": g_eff,
        "gate_selected_balance": g_bal,
        "gate_selected_tri_score": g_tri,
        "gate_source_ema_js": g_js,
        "gate_selected_coverage": g_sel,

        "gate_global_effective_classes": g_global_eff,
        "gate_global_balance": g_global_bal,

        "selected_distribution_gate_raw": selected_raw,
        "global_distribution_gate_raw": global_raw,
        "adaptation_gate_raw": combined_raw,
    }

    if not ready:
        components["adaptation_gate"] = 0.0
        components["adaptation_gate_ready"] = False
        return 0.0, components

    gate = float(
        np.clip(
            max(args.adapt_gate_floor, combined_raw),
            0.0,
            1.0,
        )
    )

    components["adaptation_gate"] = gate
    components["adaptation_gate_ready"] = True

    return gate, components



def checkpoint_is_eligible(
    diag,
    epoch_1based,
    args,
    active_adaptation_epochs=0,
    cumulative_adaptation_mass=0.0,
    controller_info=None,
):
    """
    OCRDA-v7.4.4 label-free checkpoint eligibility.

    Hard requirements are adaptation maturity + selected-distribution safety.
    Global q(c) collapse is *not* a hard rejection criterion; it is recorded as
    a warning and contributes only a soft score penalty.
    """
    controller_info = controller_info or {}
    min_epoch = (
        args.checkpoint_min_epoch
        if args.checkpoint_min_epoch > 0
        else args.warmup_epochs + 1
    )
    reasons = []
    warnings = []

    if epoch_1based < min_epoch:
        reasons.append(f"epoch<{min_epoch}")

    # Signal-based adaptation maturity, replacing the old requirement that
    # effective_ramp itself must keep growing. This allows a mature plateau
    # controller to reduce adaptation without invalidating later checkpoints.
    maturity = float(controller_info.get("v742_adaptation_maturity", 0.0))
    history_ready = bool(controller_info.get("v742_history_ready", False))
    if not history_ready:
        reasons.append("insufficient_controller_history")
    if maturity < args.v742_checkpoint_min_maturity:
        reasons.append("adaptation_not_signal_mature")
    if active_adaptation_epochs < args.checkpoint_min_active_adaptation_epochs:
        reasons.append("too_few_active_adaptation_epochs")
    if cumulative_adaptation_mass < args.checkpoint_min_cumulative_adaptation_mass:
        reasons.append("insufficient_cumulative_adaptation")

    selected_rate = float(diag.get("selected_any_rate", 0.0))
    tri_rate = float(diag.get("tri_consensus_rate", 0.0))
    selected_eff = float(diag.get("selected_effective_class_count", 0.0))
    selected_max = float(diag.get("selected_max_class_fraction", 1.0))
    tri_score = float(diag.get("mean_selected_tri_score", 0.0))
    source_ema_js = float(diag.get("mean_source_ema_js", 1.0))

    if selected_rate < args.checkpoint_min_selected_rate:
        reasons.append("selected_rate_too_low")
    if tri_rate < args.checkpoint_min_consensus_rate:
        reasons.append("tri_consensus_rate_too_low")
    if selected_eff < args.checkpoint_min_selected_effective_classes:
        reasons.append("selected_effective_class_count")
    if selected_max > args.checkpoint_max_selected_class_fraction:
        reasons.append("selected_max_class_fraction")
    if tri_score < args.checkpoint_min_selected_tri_score:
        reasons.append("selected_tri_score_too_low")
    if source_ema_js > args.checkpoint_max_source_ema_js:
        reasons.append("source_ema_js_too_high")

    # v7.4.4 hard anti-collapse gate on the *selected* adaptation samples.
    hard_collapse, hard_reasons = _hard_selected_collapse(
        diag, args.anti_collapse_min_effective_classes,
        args.anti_collapse_max_class_fraction,
    )
    if hard_collapse:
        reasons.extend(["anti_collapse:" + r for r in hard_reasons])

    # Global distribution remains diagnostic/soft safety only.
    global_eff = float(diag.get("effective_class_count_q", 0.0))
    global_max = float(diag.get("max_hard_pseudo_fraction", 1.0))
    if global_eff < args.checkpoint_min_global_effective_classes:
        warnings.append("global_effective_classes_low_soft")
    if global_max > args.checkpoint_max_global_hard_fraction:
        warnings.append("global_class_dominance_high_soft")

    return len(reasons) == 0, reasons, warnings


def resolve_split_file(split_dir, name, required=True):
    if name is None or str(name).strip() == "":
        return None
    p = Path(split_dir) / name
    if required and not p.exists():
        raise FileNotFoundError(f"Required split file not found: {p}")
    return p if p.exists() else None






def apply_method_ontology_ablation(method, rel_e, anchor_ids, topology):
    """Apply deterministic ontology ablations without target labels."""
    if method == "relation_blind":
        rel_e = torch.zeros_like(rel_e)
    elif method == "shuffled_ontology":
        # Fixed permutation, identical for every seed, to avoid introducing an
        # extra source of variance into the negative-control experiment.
        perm = torch.tensor([2, 0, 1], dtype=torch.long, device=anchor_ids.device)
        anchor_ids = anchor_ids[perm]
        topology = topology[perm][:, perm]
    elif method == "ocrda_no_topology":
        topology = torch.eye(len(CLASS_NAMES), dtype=topology.dtype, device=topology.device)
    return rel_e, anchor_ids, topology


def method_learning_snapshot(method, snapshot):
    """
    Reliability-weighting ablation: selection stays identical, while selected
    examples contribute equal weight to prototype and pseudo-label objectives.
    This isolates reliability *weighting* from tri-consensus membership.
    """
    if method != "ocrda_no_reliability":
        return snapshot
    snap = copy.deepcopy(snapshot)
    snap["tri_score"] = torch.ones_like(snap["tri_score"])
    return snap

def train_seed(args, seed, ont):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_dir = Path(args.output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    print(f"[seed={seed}] device={device}")

    source_train_tf, target_weak_tf, target_strong_tf, test_tf = build_transforms(
        args.image_size
    )
    sp = Path(args.split_dir)
    src_train_file = resolve_split_file(sp, args.source_train_file)
    src_val_file = resolve_split_file(sp, args.source_val_file)
    src_test_file = resolve_split_file(sp, args.source_test_file)
    tgt_adapt_file = resolve_split_file(sp, args.target_adapt_file)
    tgt_dev_file = (
        resolve_split_file(sp, args.target_dev_file, required=False)
        if args.target_dev_file else None
    )
    tgt_test_file = (
        resolve_split_file(
            sp, args.target_test_file, required=args.evaluate_final_test
        )
        if args.target_test_file else None
    )

    src_train = ShrimpCSVDataset(
        src_train_file, args.sdbd_root, args.tsbd_root, source_train_tf, True
    )
    src_train_proto = ShrimpCSVDataset(
        src_train_file, args.sdbd_root, args.tsbd_root, test_tf, True
    )
    src_val = ShrimpCSVDataset(
        src_val_file, args.sdbd_root, args.tsbd_root, test_tf, True
    )
    src_test = ShrimpCSVDataset(
        src_test_file, args.sdbd_root, args.tsbd_root, test_tf, True
    )
    tgt_adapt_train = ShrimpDualViewCSVDataset(
        tgt_adapt_file, args.sdbd_root, args.tsbd_root,
        target_weak_tf, target_strong_tf,
    )
    tgt_adapt_diag = ShrimpCSVDataset(
        tgt_adapt_file, args.sdbd_root, args.tsbd_root, test_tf, False
    )
    tgt_dev = (
        ShrimpCSVDataset(
            tgt_dev_file, args.sdbd_root, args.tsbd_root, test_tf, True
        ) if tgt_dev_file is not None else None
    )
    tgt_test = (
        ShrimpCSVDataset(
            tgt_test_file, args.sdbd_root, args.tsbd_root, test_tf, True
        ) if tgt_test_file is not None else None
    )

    sampler, ce_w = class_balancing(src_train)
    source_prior = dataset_class_prior(src_train, device=device)
    prior_ema = source_prior.clone()

    pin = torch.cuda.is_available()
    src_loader = DataLoader(
        src_train, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=pin, drop_last=True
    )
    src_proto_loader = DataLoader(
        src_train_proto, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin
    )
    tgt_loader = DataLoader(
        tgt_adapt_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin, drop_last=True
    )
    tgt_diag_loader = DataLoader(
        tgt_adapt_diag, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin
    )
    val_loader = DataLoader(
        src_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers
    )
    src_test_loader = DataLoader(
        src_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers
    )
    tgt_dev_loader = (
        DataLoader(tgt_dev, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.workers)
        if tgt_dev is not None else None
    )
    tgt_test_loader = (
        DataLoader(tgt_test, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.workers)
        if tgt_test is not None else None
    )

    visual = VisualModel(
        args.emb_dim,
        pretrained=not args.no_pretrained,
        dropout=args.visual_dropout,
    ).to(device)
    rgcn = RGCNOntologyEncoder(
        len(ont["nodes"]), len(ont["rel_names"]),
        args.emb_dim, args.rgcn_layers, args.dropout
    ).to(device)

    # EMA target teacher. It is reset from the source-warm student at the
    # beginning of adaptation.
    teacher_visual = copy.deepcopy(visual).to(device).eval()
    teacher_rgcn = copy.deepcopy(rgcn).to(device).eval()
    for p in teacher_visual.parameters():
        p.requires_grad_(False)
    for p in teacher_rgcn.parameters():
        p.requires_grad_(False)

    # Frozen source anchor teacher is created after source-only warm-up.
    source_teacher = None
    source_teacher_rgcn = None
    source_prototypes = None

    prototype_memory = EMAPrototypeMemory(
        num_classes=len(CLASS_NAMES),
        emb_dim=args.emb_dim,
        device=device,
        momentum=args.prototype_memory_momentum,
    )

    src_e = ont["src"].to(device)
    rel_e = ont["rel"].to(device)
    dst_e = ont["dst"].to(device)
    anchor_ids = ont["anchor_ids"].to(device)
    topology = ont["topology"].to(device)
    rel_e, anchor_ids, topology = apply_method_ontology_ablation(
        args.method, rel_e, anchor_ids, topology
    )
    if args.method == "ocrda_no_topology":
        # Disable ranking/topology objective while retaining class semantic anchors.
        args.lambda_rank = 0.0
    ce_w = ce_w.to(device)

    opt = torch.optim.AdamW(
        [
            {"params": visual.backbone.parameters(), "lr": args.backbone_lr},
            {
                "params": list(visual.proj.parameters())
                + list(visual.source_head.parameters())
                + list(visual.target_head.parameters())
                + list(rgcn.parameters()),
                "lr": args.head_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.min_lr
    )
    criterion = nn.CrossEntropyLoss(
        weight=ce_w, label_smoothing=args.label_smoothing
    )
    trainable_params = list(visual.parameters()) + list(rgcn.parameters())

    best = None
    best_score = -1e9
    strict_peak_score = -1e9

    # Secondary v7.1 recovery candidate.  It is still completely label-free:
    # the target-development metric is never used in its score or selection.
    # This prevents losing a full Kaggle run when the strict gate has no epoch.
    fallback_best = None
    fallback_best_score = -1e9
    fallback_snapshot = None

    history = []
    diagnostic_history = []
    best_snapshot = None
    prev_post_q = None

    # OCRDA-v7.4.4 adaptation maturity state.
    active_adaptation_epochs = 0
    cumulative_adaptation_mass = 0.0

    # OCRDA-v7.4.4 actual rolling plateau/drift controller state.
    v742_controller_state = {
        "next_multiplier": 1.0,
        "prev_selected_distribution": None,
        "prev_prototype_values": None,
        "prev_prototype_initialized": None,
        "post_adaptation_observations": 0,
        "plateau_counter": 0,
        "plateau_active": False,
        "plateau_age": 0,
    }
    source_retention_reference = None
    source_soft_confusion = None

    # v7.4.4 hard anti-collapse rollback + label-free checkpoint ensemble.
    last_safe_rollback_state = None
    anti_collapse_rollback_count = 0
    ensemble_candidates = []

    # v7.4.4 source-warm cache: full OCRDA is run first for each seed and saves
    # the exact pre-adaptation state. Ontology ablations then reuse that state,
    # avoiding repeated warm-up while preserving a fair common initialization.
    source_cache_path = None
    if args.source_cache_dir:
        source_cache_path = Path(args.source_cache_dir) / f"ocrda_sourcewarm_seed_{seed}.pt"
        source_cache_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if args.use_source_cache:
        if source_cache_path is None or not source_cache_path.exists():
            raise FileNotFoundError(f"Requested source cache missing: {source_cache_path}")
        cache = torch.load(source_cache_path, map_location=device)
        visual.load_state_dict(cache["visual_state_dict"])
        rgcn.load_state_dict(cache["rgcn_state_dict"])
        opt.load_state_dict(cache["optimizer_state_dict"])
        sched.load_state_dict(cache["scheduler_state_dict"])
        visual.sync_target_from_source()
        source_teacher = copy.deepcopy(visual).to(device).eval()
        source_teacher_rgcn = copy.deepcopy(rgcn).to(device).eval()
        for p in source_teacher.parameters(): p.requires_grad_(False)
        for p in source_teacher_rgcn.parameters(): p.requires_grad_(False)
        source_prototypes, source_proto_counts = compute_source_prototypes(
            source_teacher, src_proto_loader, device, head="source"
        )
        source_ref_metrics, source_ref_pred = evaluate(source_teacher, val_loader, device, head="source")
        source_soft_confusion = source_soft_confusion_from_eval(source_ref_pred, device)
        source_retention_reference = float(source_ref_metrics["macro_f1"])
        teacher_visual.load_state_dict(visual.state_dict())
        teacher_rgcn.load_state_dict(rgcn.state_dict())
        teacher_visual.eval(); teacher_rgcn.eval()
        start_epoch = args.warmup_epochs
        print(f"[seed={seed}] loaded common OCRDA source-warm cache: {source_cache_path}")

    for epoch in range(start_epoch, args.epochs):
        epoch_1based = epoch + 1
        rr = ramp(epoch, args.warmup_epochs, args.epochs)
        keep_ratio = v7_keep_ratio(
            epoch, args.warmup_epochs, args.epochs,
            args.pseudo_keep_ratio_start, args.pseudo_keep_ratio_end,
        )
        prior_source_weight = decayed_source_prior_weight(
            epoch, args.epochs,
            args.prior_source_weight_start,
            args.prior_source_weight_end,
            args.prior_source_weight_schedule,
        )

        # Freeze an independent source teacher immediately before adaptation.
        if epoch == args.warmup_epochs:
            visual.sync_target_from_source()

            source_teacher = copy.deepcopy(visual).to(device).eval()
            source_teacher_rgcn = copy.deepcopy(rgcn).to(device).eval()
            for p in source_teacher.parameters():
                p.requires_grad_(False)
            for p in source_teacher_rgcn.parameters():
                p.requires_grad_(False)

            source_prototypes, source_proto_counts = compute_source_prototypes(
                source_teacher, src_proto_loader, device, head="source"
            )

            # Fixed source-retention reference: source validation of the frozen
            # source-warm teacher immediately before target adaptation.
            source_ref_metrics, source_ref_pred = evaluate(
                source_teacher, val_loader, device, head="source"
            )
            source_soft_confusion = source_soft_confusion_from_eval(source_ref_pred, device)
            source_retention_reference = float(source_ref_metrics["macro_f1"])

            # Reset EMA target teacher from the same source-warm state.
            teacher_visual.load_state_dict(visual.state_dict())
            teacher_rgcn.load_state_dict(rgcn.state_dict())
            teacher_visual.eval()
            teacher_rgcn.eval()
            last_safe_rollback_state = {
                "visual": _cpu_fp16_state_dict(visual),
                "rgcn": _cpu_fp16_state_dict(rgcn),
                "teacher_visual": _cpu_fp16_state_dict(teacher_visual),
                "teacher_rgcn": _cpu_fp16_state_dict(teacher_rgcn),
                "prototype_values": prototype_memory.values.detach().cpu().clone(),
                "prototype_initialized": prototype_memory.initialized.detach().cpu().clone(),
                "prototype_update_counts": prototype_memory.update_counts.detach().cpu().clone(),
                "prior_ema": prior_ema.detach().cpu().clone(),
                "epoch": epoch_1based,
            }
            print(
                f"[seed={seed}] frozen source teacher created at epoch "
                f"{epoch_1based}; source prototype counts="
                f"{[int(x) for x in source_proto_counts.detach().cpu().tolist()]}; "
                f"source-retention reference F1={source_retention_reference:.4f}"
            )
            if source_cache_path is not None and not source_cache_path.exists():
                torch.save({
                    "seed": seed,
                    "visual_state_dict": visual.state_dict(),
                    "rgcn_state_dict": rgcn.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "source_retention_reference": source_retention_reference,
                    "method_version": "OCRDA-v7.4.4",
        "experiment_method": args.method,
                }, source_cache_path)
                print(f"[seed={seed}] SAVED common source-warm cache: {source_cache_path}")

        diag_source_teacher = (
            source_teacher if source_teacher is not None else teacher_visual
        )
        diag_source_rgcn = (
            source_teacher_rgcn if source_teacher_rgcn is not None else teacher_rgcn
        )

        # Before warm-up finishes, source prototypes are not used for learning;
        # ontology anchors act as a harmless diagnostic placeholder.
        diag_source_prototypes = source_prototypes

        target_prior, prior_ema, diag, snapshot = tri_consensus_diagnostics(
            source_teacher=diag_source_teacher,
            ema_teacher=teacher_visual,
            source_rgcn=diag_source_rgcn,
            loader=tgt_diag_loader,
            src_e=src_e, rel_e=rel_e, dst_e=dst_e,
            anchor_ids=anchor_ids,
            device=device,
            source_prototypes=diag_source_prototypes,
            prototype_memory=prototype_memory,
            ontology_similarity=topology,
            source_prior=source_prior,
            source_soft_confusion=source_soft_confusion,
            prior_ema=prior_ema,
            prior_ema_momentum=args.prior_ema_momentum,
            prior_source_weight=prior_source_weight,
            prior_temperature=args.prior_temperature,
            prior_floor=args.prior_floor,
            prior_ceiling=args.prior_ceiling,
            keep_ratio=keep_ratio,
            source_proto_weight=args.source_proto_weight,
            memory_proto_weight=args.memory_proto_weight,
            ontology_proto_weight=args.ontology_proto_weight,
            proto_vote_temperature=args.proto_vote_temperature,
            source_conf_threshold=args.source_conf_threshold,
            ema_conf_threshold=args.ema_conf_threshold,
            proto_conf_threshold=args.proto_conf_threshold,
            source_ema_js_max=args.source_ema_js_max,
            consensus_score_min=args.consensus_score_min,
            semantic_epsilon=args.semantic_epsilon,
            fusion_source_weight=args.fusion_source_weight,
            fusion_ema_weight=args.fusion_ema_weight,
            fusion_proto_weight=args.fusion_proto_weight,
            collapse_threshold=args.collapse_threshold,
            prior_evidence_weight_consensus=args.prior_evidence_weight_consensus,
            prior_evidence_weight_teacher_median=args.prior_evidence_weight_teacher_median,
            prior_evidence_weight_fusion=args.prior_evidence_weight_fusion,
            prior_evidence_weight_bbse=args.prior_evidence_weight_bbse,
            quota_temper_gamma=args.quota_temper_gamma,
            quota_min_class_fraction=args.quota_min_class_fraction,
            quota_min_class_count=args.quota_min_class_count,
        )

        # Target memory is updated only from globally selected tri-consensus
        # candidates, never from generic high-confidence pseudo-labels.
        learning_snapshot = method_learning_snapshot(args.method, snapshot)
        if epoch >= args.warmup_epochs:
            protos, valid, masses = consensus_prototypes_from_snapshot(
                learning_snapshot, min_selected=args.prototype_global_min_selected
            )
            prototype_memory.update_many(
                protos, valid, momentum=args.prototype_epoch_momentum
            )
        else:
            masses = torch.zeros(len(CLASS_NAMES))

        mem_diag = prototype_memory.summary()
        diag["prototype_memory_initialized_classes"] = mem_diag["initialized_classes"]
        for c, name in enumerate(CLASS_NAMES):
            diag[f"prototype_memory_updates_{name}"] = mem_diag["update_counts"][c]
            diag[f"prototype_selected_mass_{name}"] = float(masses[c].item())

        pseudo_cache = build_consensus_cache(
            learning_snapshot, topology, semantic_epsilon=args.semantic_epsilon
        )

        adaptation_gate, gate_diag = selected_distribution_adaptation_gate(
            diag, args
        )
        diag.update(gate_diag)
        base_adaptation_weight = rr * adaptation_gate
        controller_multiplier_used = float(
            v742_controller_state.get("next_multiplier", 1.0)
        )
        effective_ramp = base_adaptation_weight * controller_multiplier_used

        cumulative_adaptation_mass += max(
            float(effective_ramp), 0.0
        )

        if (
            effective_ramp
            >= args.maturity_active_ramp_threshold
        ):
            active_adaptation_epochs += 1

        diag["base_ocrda_ramp"] = float(rr)
        diag["v742_base_adaptation_weight"] = float(base_adaptation_weight)
        diag["v742_controller_multiplier_used"] = float(controller_multiplier_used)
        diag["effective_ocrda_ramp"] = float(effective_ramp)
        diag["v742_effective_adaptation_weight"] = float(effective_ramp)
        diag["active_adaptation_epochs"] = int(
            active_adaptation_epochs
        )
        diag["cumulative_adaptation_mass"] = float(
            cumulative_adaptation_mass
        )

        # Log current class-balanced semantic weights.
        _cb = class_balance_weights_from_prior(
            target_prior,
            gamma=args.class_balance_gamma,
            min_weight=args.class_balance_min,
            max_weight=args.class_balance_max,
        )
        for _ci, _cn in enumerate(CLASS_NAMES):
            diag[f"class_balance_weight_{_cn}"] = float(
                _cb[_ci].detach().cpu()
            )
        diag["global_collapse_warning_only"] = bool(diag.get("collapse_flag", False))

        diagnostic_history.append({"epoch": epoch_1based, **diag})

        print(
            f"[seed={seed}] pre-epoch={epoch_1based:03d} "
            f"q=[{diag['q_mean_BG']:.3f},{diag['q_mean_Healthy']:.3f},"
            f"{diag['q_mean_WSSV']:.3f}] "
            f"tri={diag['tri_consensus_rate']:.3f} "
            f"eligible={diag['tri_eligible_rate']:.3f} "
            f"selected={diag['selected_any_rate']:.3f} "
            f"selEff={diag['selected_effective_class_count']:.2f} "
            f"selMax={diag['selected_max_class_fraction']:.2f} "
            f"score={diag['mean_selected_tri_score']:.3f} "
            f"JS={diag['mean_source_ema_js']:.3f} "
            f"gate={adaptation_gate:.3f} "
            f"ramp={rr:.3f} base={base_adaptation_weight:.3f} "
            f"ctrl={controller_multiplier_used:.3f} -> eff={effective_ramp:.3f} "
            f"memC={mem_diag['initialized_classes']} "
            f"globalCollapse={diag['collapse_flag']}"
        )

        visual.train()
        rgcn.train()
        teacher_visual.eval()
        teacher_rgcn.eval()
        if source_teacher is not None:
            source_teacher.eval()
            source_teacher_rgcn.eval()

        tgt_iter = iter(tgt_loader)
        total_loss = 0.0
        batches = 0
        stat_sums = defaultdict(float)

        for step_idx, (xs, ys, _) in enumerate(src_loader):
            if (
                epoch >= args.warmup_epochs
                and args.max_adapt_steps_per_epoch > 0
                and step_idx >= args.max_adapt_steps_per_epoch
            ):
                break
            try:
                xt_w, xt_s, _, sid_t = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                xt_w, xt_s, _, sid_t = next(tgt_iter)

            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            xt_w = xt_w.to(device, non_blocking=True)
            xt_s = xt_s.to(device, non_blocking=True)

            zs, logits_source_s, logits_target_on_source = visual(
                xs, return_both=True
            )
            zt_s, _, logits_target_s = visual(xt_s, return_both=True)
            H_student = rgcn(src_e, rel_e, dst_e)
            anchors_student = H_student[anchor_ids]

            l_source = criterion(logits_source_s, ys)
            l_target_source_anchor = criterion(logits_target_on_source, ys)
            l_sem = semantic_alignment_loss(
                zs, ys, anchors_student, args.tau_sem
            )

            if effective_ramp > 0 and source_teacher is not None:
                with torch.no_grad():
                    _, logits_ema_t = teacher_visual(xt_w, head="target")
                    _, logits_source_t = source_teacher(xt_w, head="source")

                (
                    l_soft,
                    l_ema_cons,
                    l_source_cons,
                    l_proto,
                    target_stats,
                    batch_selected,
                    batch_hard,
                    batch_weight,
                ) = consensus_target_losses(
                    logits_target_student=logits_target_s,
                    z_target_student=zt_s,
                    logits_target_teacher=logits_ema_t,
                    logits_source_teacher=logits_source_t,
                    sample_ids=sid_t,
                    pseudo_cache=pseudo_cache,
                    student_anchors=anchors_student,
                    source_prototypes=source_prototypes,
                    prototype_memory=prototype_memory,
                    ontology_similarity=topology,
                    source_proto_weight=args.source_proto_weight,
                    memory_proto_weight=args.memory_proto_weight,
                    ontology_proto_weight=args.ontology_proto_weight,
                    proto_temperature=args.proto_temperature,
                    target_prior=target_prior,
                    class_balance_gamma=args.class_balance_gamma,
                    class_balance_min=args.class_balance_min,
                    class_balance_max=args.class_balance_max,
                )

                l_mean, l_rank, topo_stats = v7_mean_topology_loss(
                    zs=zs,
                    ys=ys,
                    zt=zt_s,
                    selected=batch_selected,
                    hard=batch_hard,
                    sample_weight=batch_weight,
                    student_anchors=anchors_student,
                    source_prototypes=source_prototypes,
                    prototype_memory=prototype_memory,
                    ontology_similarity=topology,
                    target_prior=target_prior,
                    source_proto_weight=args.source_proto_weight,
                    memory_proto_weight=args.memory_proto_weight,
                    ontology_proto_weight=args.ontology_proto_weight,
                    current_target_weight=args.prototype_current_weight,
                    rank_margin=args.rank_margin,
                    rank_temperature=args.rank_temperature,
                    topology_similarity_delta=args.topology_similarity_delta,
                )
            else:
                zero = logits_target_s.sum() * 0.0
                l_soft = l_ema_cons = l_source_cons = l_proto = zero
                l_mean = l_rank = zero
                target_stats = {
                    "selected_pseudo_rate_batch": 0.0,
                    "mean_selected_weight_batch": 0.0,
                    "soft_pseudo_ce": 0.0,
                    "ema_consistency": 0.0,
                    "source_teacher_consistency": 0.0,
                    "prototype_contrastive": 0.0,
                }
                topo_stats = {
                    "mean_alignment": 0.0,
                    "ranking_topology": 0.0,
                    "topology_gradient_classes_batch": 0,
                    "topology_pairs_batch": 0,
                    "topology_pairs_with_current_gradient_batch": 0,
                    "prototype_memory_initialized_classes":
                        int(prototype_memory.initialized.sum().item()),
                }

            adaptation_loss = (
                args.lambda_soft_pseudo * l_soft
                + args.lambda_teacher_consistency * l_ema_cons
                + args.lambda_source_teacher_consistency * l_source_cons
                + args.lambda_proto_contrast * l_proto
                + args.lambda_mean_alignment * l_mean
                + args.lambda_rank * l_rank
            )

            loss = (
                l_source
                + args.lambda_target_source_anchor * l_target_source_anchor
                + args.lambda_sem * l_sem
                + effective_ramp * adaptation_loss
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            opt.step()

            ema_update_module(
                teacher_visual, visual, momentum=args.teacher_ema
            )
            ema_update_module(
                teacher_rgcn, rgcn, momentum=args.teacher_ema
            )

            total_loss += float(loss.detach().cpu())
            batches += 1
            stat_sums["source_ce"] += float(l_source.detach().cpu())
            stat_sums["target_source_anchor_ce"] += float(
                l_target_source_anchor.detach().cpu()
            )
            stat_sums["sem"] += float(l_sem.detach().cpu())
            stat_sums["soft_pseudo"] += float(l_soft.detach().cpu())
            stat_sums["ema_cons"] += float(l_ema_cons.detach().cpu())
            stat_sums["source_cons"] += float(l_source_cons.detach().cpu())
            stat_sums["proto"] += float(l_proto.detach().cpu())
            stat_sums["mean_align"] += float(l_mean.detach().cpu())
            stat_sums["rank"] += float(l_rank.detach().cpu())

            for k, v in target_stats.items():
                stat_sums[k] += float(v)
            for k, v in topo_stats.items():
                stat_sums[k] += float(v)

        sched.step()

        # Source validation uses the source head of the EMA model.
        val_metrics, _ = evaluate(
            teacher_visual, val_loader, device, head="source"
        )

        # Post-epoch label-free target diagnostics use the frozen source teacher
        # and current EMA target teacher. No target labels are consulted.
        if source_teacher is not None:
            post_target_prior, _, post_diag, post_snapshot = tri_consensus_diagnostics(
                source_teacher=source_teacher,
                ema_teacher=teacher_visual,
                source_rgcn=source_teacher_rgcn,
                loader=tgt_diag_loader,
                src_e=src_e, rel_e=rel_e, dst_e=dst_e,
                anchor_ids=anchor_ids,
                device=device,
                source_prototypes=source_prototypes,
                prototype_memory=prototype_memory,
                ontology_similarity=topology,
                source_prior=source_prior,
                source_soft_confusion=source_soft_confusion,
                prior_ema=prior_ema,
                prior_ema_momentum=args.prior_ema_momentum,
                prior_source_weight=prior_source_weight,
                prior_temperature=args.prior_temperature,
                prior_floor=args.prior_floor,
                prior_ceiling=args.prior_ceiling,
                keep_ratio=keep_ratio,
                source_proto_weight=args.source_proto_weight,
                memory_proto_weight=args.memory_proto_weight,
                ontology_proto_weight=args.ontology_proto_weight,
                proto_vote_temperature=args.proto_vote_temperature,
                source_conf_threshold=args.source_conf_threshold,
                ema_conf_threshold=args.ema_conf_threshold,
                proto_conf_threshold=args.proto_conf_threshold,
                source_ema_js_max=args.source_ema_js_max,
                consensus_score_min=args.consensus_score_min,
                semantic_epsilon=args.semantic_epsilon,
                fusion_source_weight=args.fusion_source_weight,
                fusion_ema_weight=args.fusion_ema_weight,
                fusion_proto_weight=args.fusion_proto_weight,
                collapse_threshold=args.collapse_threshold,
                prior_evidence_weight_consensus=args.prior_evidence_weight_consensus,
                prior_evidence_weight_teacher_median=args.prior_evidence_weight_teacher_median,
                prior_evidence_weight_fusion=args.prior_evidence_weight_fusion,
                prior_evidence_weight_bbse=args.prior_evidence_weight_bbse,
                quota_temper_gamma=args.quota_temper_gamma,
                quota_min_class_fraction=args.quota_min_class_fraction,
                quota_min_class_count=args.quota_min_class_count,
            )
        else:
            post_target_prior = target_prior
            post_diag = dict(diag)
            post_snapshot = snapshot

        post_q = torch.tensor(
            [post_diag[f"q_mean_{c}"] for c in CLASS_NAMES],
            dtype=torch.float32,
        )
        q_drift = (
            float(torch.abs(post_q - prev_post_q).sum().item())
            if prev_post_q is not None else 0.0
        )
        prev_post_q = post_q

        # Update the actual rolling controller from post-epoch, label-free
        # signals. The resulting multiplier is applied only to the NEXT epoch.
        controller_info = v742_update_controller(
            state=v742_controller_state,
            post_diag=post_diag,
            prototype_memory=prototype_memory,
            teacher_rgcn=teacher_rgcn,
            source_teacher_rgcn=source_teacher_rgcn,
            src_e=src_e, rel_e=rel_e, dst_e=dst_e, anchor_ids=anchor_ids,
            source_val_macro_f1=val_metrics["macro_f1"],
            source_retention_reference=source_retention_reference,
            args=args,
            adaptation_active=(effective_ramp >= args.maturity_active_ramp_threshold),
        )

        checkpoint_eligible, checkpoint_reasons, checkpoint_warnings = checkpoint_is_eligible(
            post_diag,
            epoch_1based,
            args,
            active_adaptation_epochs=active_adaptation_epochs,
            cumulative_adaptation_mass=cumulative_adaptation_mass,
            controller_info=controller_info,
        )
        anti_collapse_triggered, anti_collapse_reasons = _hard_selected_collapse(
            post_diag, args.anti_collapse_min_effective_classes,
            args.anti_collapse_max_class_fraction,
        )

        # Label-free OCRDA-v7.4.4 checkpoint score. It rewards source
        # retention, tri-consensus quality, actual adaptation maturity and
        # selected-distribution stability. Global q(c) safety is a soft term.
        consensus_quality = float(post_diag.get("mean_selected_tri_score", 0.0))
        js_term = float(post_diag.get("mean_source_ema_js", 1.0))
        maturity_score = float(controller_info.get("v742_adaptation_maturity", 0.0))
        selected_stability_score = float(
            controller_info.get("v742_selected_distribution_stability", 0.0)
        )
        source_retention_trust = float(
            controller_info.get("v742_source_retention_trust", 1.0)
        )
        anchor_drift = controller_info.get("v742_source_anchor_drift", np.nan)
        anchor_drift_term = (
            float(anchor_drift) if np.isfinite(anchor_drift) else 0.0
        )

        global_eff_score = _gate_high_good(
            float(post_diag.get("effective_class_count_q", 0.0)),
            args.dual_gate_global_eff_low,
            args.dual_gate_global_eff_high,
        )
        global_bal_score = _gate_low_good(
            float(post_diag.get("max_hard_pseudo_fraction", 1.0)),
            args.dual_gate_global_maxfrac_good,
            args.dual_gate_global_maxfrac_bad,
        )
        global_safety_score = 0.5 * global_eff_score + 0.5 * global_bal_score

        source_score_weight = min(
            args.checkpoint_source_val_weight,
            args.checkpoint_source_val_weight_cap,
        )

        checkpoint_score = (
            source_score_weight * val_metrics["macro_f1"]
            + args.checkpoint_consensus_weight * consensus_quality
            + args.checkpoint_maturity_weight * maturity_score
            + args.v742_checkpoint_selected_stability_weight * selected_stability_score
            + args.v742_checkpoint_source_retention_weight * source_retention_trust
            + args.checkpoint_global_safety_weight * global_safety_score
            - args.checkpoint_js_weight * js_term
            - args.checkpoint_drift_weight * q_drift
            - args.v742_checkpoint_anchor_drift_weight * anchor_drift_term
        )

        target_dev_metric = None
        if args.report_target_dev and tgt_dev_loader is not None:
            target_dev_metric, _ = evaluate(
                teacher_visual, tgt_dev_loader, device, head="target"
            )

        denom = max(batches, 1)
        rec = {
            "epoch": epoch_1based,
            "loss": total_loss / denom,
            "source_val_macro_f1": val_metrics["macro_f1"],
            "target_dev_macro_f1": (
                target_dev_metric["macro_f1"]
                if target_dev_metric is not None else np.nan
            ),
            "base_ocrda_ramp": rr,
            "adaptation_gate": adaptation_gate,
            "adaptation_gate_raw": gate_diag.get("adaptation_gate_raw", 0.0),
            "gate_selected_effective_classes": gate_diag.get("gate_selected_effective_classes", 0.0),
            "gate_selected_balance": gate_diag.get("gate_selected_balance", 0.0),
            "gate_selected_tri_score": gate_diag.get("gate_selected_tri_score", 0.0),
            "gate_source_ema_js": gate_diag.get("gate_source_ema_js", 0.0),
            "gate_selected_coverage": gate_diag.get("gate_selected_coverage", 0.0),
            "effective_ocrda_ramp": effective_ramp,
            "v742_base_adaptation_weight": base_adaptation_weight,
            "v742_controller_multiplier_used": controller_multiplier_used,
            "v742_effective_adaptation_weight": effective_ramp,
            "v742_next_controller_multiplier": controller_info.get("v742_next_controller_multiplier", 1.0),
            "v742_selected_tri_score": controller_info.get("v742_selected_tri_score", np.nan),
            "v742_source_ema_js": controller_info.get("v742_source_ema_js", np.nan),
            "v742_selected_distribution_drift": controller_info.get("v742_selected_distribution_drift", np.nan),
            "v742_prototype_drift": controller_info.get("v742_prototype_drift", np.nan),
            "v742_source_anchor_drift": controller_info.get("v742_source_anchor_drift", np.nan),
            "v742_source_retention_drop": controller_info.get("v742_source_retention_drop", np.nan),
            "v742_selected_distribution_stability": controller_info.get("v742_selected_distribution_stability", 0.0),
            "v742_prototype_stability": controller_info.get("v742_prototype_stability", 0.0),
            "v742_source_anchor_stability": controller_info.get("v742_source_anchor_stability", 0.0),
            "v742_adaptation_maturity": controller_info.get("v742_adaptation_maturity", 0.0),
            "v742_history_ready": controller_info.get("v742_history_ready", False),
            "v742_plateau_counter": controller_info.get("v742_plateau_counter", 0),
            "v742_plateau_age": controller_info.get("v742_plateau_age", 0),
            "v742_plateau_factor": controller_info.get("v742_plateau_factor", 1.0),
            "v742_plateau_active": controller_info.get("v742_plateau_active", False),
            "v742_source_retention_trust": controller_info.get("v742_source_retention_trust", 1.0),
            "v742_drift_guard": controller_info.get("v742_drift_guard", 1.0),
            "checkpoint_soft_warnings": "|".join(checkpoint_warnings),
            "anti_collapse_triggered": bool(anti_collapse_triggered),
            "anti_collapse_reasons": "|".join(anti_collapse_reasons),
            "anti_collapse_rollback_count_before_epoch_end": int(anti_collapse_rollback_count),
            "pseudo_keep_ratio_used": keep_ratio,
            "prior_source_weight_scheduled": prior_source_weight,
            "prior_source_weight_used": post_diag.get("prior_source_weight_used", prior_source_weight),
            "prior_source_js": post_diag.get("prior_source_js", np.nan),
            "checkpoint_score": checkpoint_score,
            "checkpoint_eligible": bool(checkpoint_eligible),
            "checkpoint_ineligibility_reasons": "|".join(checkpoint_reasons),
            "checkpoint_effective_class_count_q":
                post_diag["effective_class_count_q"],
            "checkpoint_max_hard_pseudo_fraction":
                post_diag["max_hard_pseudo_fraction"],
            "checkpoint_selected_effective_class_count":
                post_diag.get("selected_effective_class_count", np.nan),
            "checkpoint_selected_max_class_fraction":
                post_diag.get("selected_max_class_fraction", np.nan),
            "checkpoint_selected_total_count":
                post_diag.get("selected_total_count", 0),
            "checkpoint_q_l1_drift": q_drift,
            "post_tri_consensus_rate": post_diag.get("tri_consensus_rate", 0.0),
            "post_selected_any_rate": post_diag.get("selected_any_rate", 0.0),
            "post_mean_selected_tri_score":
                post_diag.get("mean_selected_tri_score", 0.0),
            "post_mean_source_ema_js":
                post_diag.get("mean_source_ema_js", 0.0),
            "mean_source_ce_loss": stat_sums["source_ce"] / denom,
            "mean_target_source_anchor_ce_loss":
                stat_sums["target_source_anchor_ce"] / denom,
            "mean_soft_pseudo_ce_loss": stat_sums["soft_pseudo"] / denom,
            "mean_ema_consistency_loss": stat_sums["ema_cons"] / denom,
            "mean_source_teacher_consistency_loss":
                stat_sums["source_cons"] / denom,
            "mean_prototype_contrastive_loss": stat_sums["proto"] / denom,
            "mean_cosine_alignment_loss": stat_sums["mean_align"] / denom,
            "mean_ranking_topology_loss": stat_sums["rank"] / denom,
            "mean_selected_pseudo_rate_batch":
                stat_sums["selected_pseudo_rate_batch"] / denom,
            "mean_selected_weight_batch":
                stat_sums["mean_selected_weight_batch"] / denom,
            "mean_topology_gradient_classes_batch":
                stat_sums["topology_gradient_classes_batch"] / denom,
            "mean_topology_pairs_with_current_gradient_batch":
                stat_sums["topology_pairs_with_current_gradient_batch"] / denom,
            "mean_prototype_memory_initialized_classes":
                stat_sums["prototype_memory_initialized_classes"] / denom,
        }
        for c in CLASS_NAMES:
            rec[f"post_q_mean_{c}"] = post_diag.get(f"q_mean_{c}", np.nan)
            rec[f"post_target_prior_{c}"] = post_diag.get(
                f"target_prior_{c}", np.nan
            )
            rec[f"post_sampling_prior_{c}"] = post_diag.get(
                f"sampling_prior_{c}", np.nan
            )
        history.append(rec)

        msg = (
            f"[seed={seed}] epoch={epoch_1based:03d} "
            f"loss={rec['loss']:.4f} "
            f"srcValF1={val_metrics['macro_f1']:.4f} "
            f"tri={post_diag.get('tri_consensus_rate',0):.3f} "
            f"selected={post_diag.get('selected_any_rate',0):.3f} "
            f"triScore={post_diag.get('mean_selected_tri_score',0):.3f} "
            f"JS={post_diag.get('mean_source_ema_js',0):.3f} "
            f"ckptScore={checkpoint_score:.4f} "
            f"selEff={post_diag.get('selected_effective_class_count',0):.2f} "
            f"selMax={post_diag.get('selected_max_class_fraction',1):.2f} "
            f"eligible={checkpoint_eligible} "
            f"maturity={controller_info.get('v742_adaptation_maturity',0):.3f} "
            f"plateau={int(controller_info.get('v742_plateau_active',False))} "
            f"nextCtrl={controller_info.get('v742_next_controller_multiplier',1):.3f} "
            f"antiCollapse={int(anti_collapse_triggered)}"
        )
        if not checkpoint_eligible:
            msg += f" gate={'|'.join(checkpoint_reasons)}"
        if checkpoint_warnings:
            msg += f" softWarn={'|'.join(checkpoint_warnings)}"
        if target_dev_metric is not None:
            msg += f" targetDevF1={target_dev_metric['macro_f1']:.4f}"
        print(msg)

        # Track the highest-scoring post-warmup label-free checkpoint
        # independently of the strict gate. Unlike v7.3, the fallback is not
        # silently constrained by a large effective-ramp threshold.
        fallback_candidate = (
            epoch_1based >= max(args.checkpoint_min_epoch, args.warmup_epochs + 1)
            and active_adaptation_epochs >= args.v742_fallback_min_active_epochs
            and np.isfinite(checkpoint_score)
            and post_diag.get("tri_consensus_rate", 0.0) > 0.0
            and post_diag.get("selected_any_rate", 0.0) > 0.0
            and not anti_collapse_triggered
        )
        if fallback_candidate:
            if checkpoint_score > fallback_best_score + args.min_delta:
                fallback_best_score = checkpoint_score
                choose_fallback = True
            else:
                choose_fallback = (
                    checkpoint_score >= fallback_best_score - args.checkpoint_late_tolerance_abs
                )
        else:
            choose_fallback = False
        if choose_fallback:
            fallback_best = {
                "epoch": epoch_1based,
                "checkpoint_score": float(checkpoint_score),
                "source_val": val_metrics,
                "visual": copy.deepcopy(teacher_visual.state_dict()),
                "rgcn": copy.deepcopy(teacher_rgcn.state_dict()),
                "student_visual": copy.deepcopy(visual.state_dict()),
                "student_rgcn": copy.deepcopy(rgcn.state_dict()),
                "pseudo_diagnostics": copy.deepcopy(post_diag),
                "prototype_memory_values":
                    prototype_memory.values.detach().cpu().clone(),
                "prototype_memory_initialized":
                    prototype_memory.initialized.detach().cpu().clone(),
                "prototype_memory_update_counts":
                    prototype_memory.update_counts.detach().cpu().clone(),
                "source_prototypes":
                    source_prototypes.detach().cpu().clone()
                    if source_prototypes is not None else None,
                "checkpoint_fallback_used": True,
                "checkpoint_fallback_policy":
                    "latest label-free tri-consensus checkpoint within tolerance "
                    "of the peak score, never earlier than checkpoint_min_epoch, "
                    "when the strict selected-distribution gate has no eligible epoch",
                "strict_gate_reasons_at_selected_epoch":
                    list(checkpoint_reasons),
                "soft_gate_warnings_at_selected_epoch": list(checkpoint_warnings),
                "controller_info": copy.deepcopy(controller_info),
            }
            fallback_snapshot = copy.deepcopy(post_snapshot)

        if checkpoint_eligible:
            if checkpoint_score > strict_peak_score + args.min_delta:
                strict_peak_score = checkpoint_score
                choose_strict = True
            else:
                choose_strict = (
                    checkpoint_score >= strict_peak_score - args.checkpoint_late_tolerance_abs
                )
        else:
            choose_strict = False
        if choose_strict:
            best_score = checkpoint_score
            best = {
                "epoch": epoch_1based,
                "checkpoint_score": float(checkpoint_score),
                "source_val": val_metrics,
                "visual": copy.deepcopy(teacher_visual.state_dict()),
                "rgcn": copy.deepcopy(teacher_rgcn.state_dict()),
                "student_visual": copy.deepcopy(visual.state_dict()),
                "student_rgcn": copy.deepcopy(rgcn.state_dict()),
                "pseudo_diagnostics": copy.deepcopy(post_diag),
                "prototype_memory_values":
                    prototype_memory.values.detach().cpu().clone(),
                "prototype_memory_initialized":
                    prototype_memory.initialized.detach().cpu().clone(),
                "prototype_memory_update_counts":
                    prototype_memory.update_counts.detach().cpu().clone(),
                "source_prototypes":
                    source_prototypes.detach().cpu().clone()
                    if source_prototypes is not None else None,
                "checkpoint_fallback_used": False,
                "soft_gate_warnings_at_selected_epoch": list(checkpoint_warnings),
                "controller_info": copy.deepcopy(controller_info),
            }
            best_snapshot = copy.deepcopy(post_snapshot)

        # v7.4.4 label-free safe-checkpoint ensemble pool.  Only strict,
        # anti-collapse-safe checkpoints are admitted; target-dev is never used.
        if checkpoint_eligible and not anti_collapse_triggered and np.isfinite(checkpoint_score):
            ensemble_candidates.append({
                "epoch": int(epoch_1based),
                "score": float(checkpoint_score),
                "visual": _cpu_fp16_state_dict(teacher_visual),
                "rgcn": _cpu_fp16_state_dict(teacher_rgcn),
                "source_val": copy.deepcopy(val_metrics),
                "pseudo_diagnostics": copy.deepcopy(post_diag),
                "prototype_memory_values": prototype_memory.values.detach().cpu().clone(),
                "prototype_memory_initialized": prototype_memory.initialized.detach().cpu().clone(),
                "prototype_memory_update_counts": prototype_memory.update_counts.detach().cpu().clone(),
                "source_prototypes": source_prototypes.detach().cpu().clone() if source_prototypes is not None else None,
                "controller_info": copy.deepcopy(controller_info),
                "snapshot": copy.deepcopy(post_snapshot),
            })
            peak = max(x["score"] for x in ensemble_candidates)
            ensemble_candidates = [
                x for x in ensemble_candidates
                if x["score"] >= peak - args.ensemble_pool_tolerance_abs
            ]
            # Keep a bounded, high-score pool to avoid unnecessary host RAM use.
            ensemble_candidates = sorted(
                ensemble_candidates, key=lambda x: (x["score"], x["epoch"]), reverse=True
            )[:args.ensemble_max_pool]

        # v7.4.4 anti-collapse rollback affects the NEXT epoch only.  The bad
        # epoch remains in diagnostics but cannot be selected as a checkpoint.
        if anti_collapse_triggered:
            if last_safe_rollback_state is not None:
                visual.load_state_dict(last_safe_rollback_state["visual"])
                rgcn.load_state_dict(last_safe_rollback_state["rgcn"])
                teacher_visual.load_state_dict(last_safe_rollback_state["teacher_visual"])
                teacher_rgcn.load_state_dict(last_safe_rollback_state["teacher_rgcn"])
                prototype_memory.values.copy_(last_safe_rollback_state["prototype_values"].to(device))
                prototype_memory.initialized.copy_(last_safe_rollback_state["prototype_initialized"].to(device))
                prototype_memory.update_counts.copy_(last_safe_rollback_state["prototype_update_counts"].to(device))
                prior_ema = last_safe_rollback_state["prior_ema"].to(device)
                # Clear Adam moments from the collapsed excursion; source-warm
                # learning rates/scheduler remain unchanged.
                opt.state.clear()
                anti_collapse_rollback_count += 1
                v742_controller_state["next_multiplier"] = min(
                    float(v742_controller_state.get("next_multiplier", 1.0)),
                    float(args.anti_collapse_rollback_multiplier),
                )
                print(
                    f"[seed={seed}] ANTI-COLLAPSE ROLLBACK -> safe epoch "
                    f"{last_safe_rollback_state.get('epoch')} reasons={anti_collapse_reasons}; "
                    f"next multiplier <= {args.anti_collapse_rollback_multiplier:.3f}",
                    flush=True,
                )
        else:
            last_safe_rollback_state = {
                "visual": _cpu_fp16_state_dict(visual),
                "rgcn": _cpu_fp16_state_dict(rgcn),
                "teacher_visual": _cpu_fp16_state_dict(teacher_visual),
                "teacher_rgcn": _cpu_fp16_state_dict(teacher_rgcn),
                "prototype_values": prototype_memory.values.detach().cpu().clone(),
                "prototype_initialized": prototype_memory.initialized.detach().cpu().clone(),
                "prototype_update_counts": prototype_memory.update_counts.detach().cpu().clone(),
                "prior_ema": prior_ema.detach().cpu().clone(),
                "epoch": epoch_1based,
            }

    if best is None:
        if fallback_best is None:
            # Robust experiment-suite behavior: record a genuine adaptation
            # failure instead of aborting the remaining methods/seeds. The
            # fallback is the frozen source-warm teacher and uses NO target
            # labels. This makes collapsed ablations measurable rather than
            # silently dropping them from the comparison.
            if source_teacher is None:
                raise RuntimeError("No source-warm teacher available for failure fallback")
            fallback_best = {
                "epoch": int(args.warmup_epochs),
                "checkpoint_score": float(source_retention_reference or 0.0),
                "source_val": source_ref_metrics if "source_ref_metrics" in locals() else {},
                "visual": copy.deepcopy(source_teacher.state_dict()),
                "rgcn": copy.deepcopy(source_teacher_rgcn.state_dict()),
                "student_visual": copy.deepcopy(visual.state_dict()),
                "student_rgcn": copy.deepcopy(rgcn.state_dict()),
                "pseudo_diagnostics": copy.deepcopy(post_diag) if "post_diag" in locals() else {},
                "prototype_memory_values": prototype_memory.values.detach().cpu().clone(),
                "prototype_memory_initialized": prototype_memory.initialized.detach().cpu().clone(),
                "prototype_memory_update_counts": prototype_memory.update_counts.detach().cpu().clone(),
                "source_prototypes": source_prototypes.detach().cpu().clone() if source_prototypes is not None else None,
                "checkpoint_fallback_used": True,
                "fallback_reason": "no_postwarm_selected_tri_consensus; source-warm fallback",
                "strict_gate_reasons_at_selected_epoch": ["adaptation_failure_no_selected_candidates"],
                "soft_gate_warnings_at_selected_epoch": [],
                "controller_info": {},
            }
            fallback_snapshot = copy.deepcopy(snapshot) if "snapshot" in locals() else None
            fallback_best_score = float(fallback_best["checkpoint_score"])
        best = fallback_best
        best_snapshot = fallback_snapshot
        best_score = fallback_best_score
        print(
            "[OCRDA-v7.4.4] WARNING: no epoch satisfied the strict selected-"
            "distribution safety gate. Using the latest LABEL-FREE checkpoint "
            "within tolerance of the post-min-epoch tri-consensus peak. "
            f"Selected epoch={best['epoch']}, score={best_score:.4f}, "
            f"strict reasons={best.get('strict_gate_reasons_at_selected_epoch')}"
        )

    # v7.4.4 final model = label-free SWA-style average of up to top-K
    # anti-collapse-safe checkpoints within a score tolerance of the peak.
    ensemble_used = False
    ensemble_epochs = []
    ensemble_scores = []
    if len(ensemble_candidates) >= int(args.ensemble_min_count):
        peak = max(x["score"] for x in ensemble_candidates)
        cand = [
            x for x in ensemble_candidates
            if x["score"] >= peak - args.ensemble_score_tolerance_abs
        ]
        cand = sorted(cand, key=lambda x: (x["score"], x["epoch"]), reverse=True)
        cand = cand[:int(args.ensemble_top_k)]
        if len(cand) >= int(args.ensemble_min_count):
            avg_visual = _average_state_dicts([x["visual"] for x in cand])
            avg_rgcn = _average_state_dicts([x["rgcn"] for x in cand])
            representative = max(cand, key=lambda x: x["score"])
            latest = max(cand, key=lambda x: x["epoch"])
            diag_rep = copy.deepcopy(representative["pseudo_diagnostics"])
            ensemble_epochs = sorted([int(x["epoch"]) for x in cand])
            ensemble_scores = [float(x["score"]) for x in cand]
            diag_rep["checkpoint_ensemble_size"] = len(cand)
            diag_rep["checkpoint_ensemble_epochs"] = ensemble_epochs
            diag_rep["checkpoint_ensemble_scores"] = ensemble_scores
            diag_rep["anti_collapse_rollback_count"] = int(anti_collapse_rollback_count)
            best = {
                "epoch": int(latest["epoch"]),
                "checkpoint_score": float(peak),
                "source_val": representative["source_val"],
                "visual": avg_visual,
                "rgcn": avg_rgcn,
                "student_visual": avg_visual,
                "student_rgcn": avg_rgcn,
                "pseudo_diagnostics": diag_rep,
                "prototype_memory_values": latest["prototype_memory_values"],
                "prototype_memory_initialized": latest["prototype_memory_initialized"],
                "prototype_memory_update_counts": latest["prototype_memory_update_counts"],
                "source_prototypes": latest["source_prototypes"],
                "checkpoint_fallback_used": False,
                "soft_gate_warnings_at_selected_epoch": [],
                "controller_info": representative["controller_info"],
                "checkpoint_ensemble_used": True,
                "checkpoint_ensemble_epochs": ensemble_epochs,
                "checkpoint_ensemble_scores": ensemble_scores,
            }
            best_snapshot = representative["snapshot"]
            best_score = float(peak)
            ensemble_used = True
            print(
                f"[OCRDA-v7.4.4] SAFE CHECKPOINT ENSEMBLE: epochs={ensemble_epochs}, "
                f"scores={[round(x,4) for x in ensemble_scores]}", flush=True
            )

    teacher_visual.load_state_dict(best["visual"])
    teacher_rgcn.load_state_dict(best["rgcn"])

    final_source_val_metrics, _ = evaluate(
        teacher_visual, val_loader, device, head="source"
    )
    src_test_metrics, _ = evaluate(
        teacher_visual, src_test_loader, device, head="source"
    )
    target_dev_metrics = target_dev_pred = None
    if tgt_dev_loader is not None:
        target_dev_metrics, target_dev_pred = evaluate(
            teacher_visual, tgt_dev_loader, device, head="target"
        )

    target_test_metrics = target_test_pred = None
    if args.evaluate_final_test:
        if tgt_test_loader is None:
            raise FileNotFoundError(
                "Final target evaluation requested but target-test file unavailable."
            )
        target_test_metrics, target_test_pred = evaluate(
            teacher_visual, tgt_test_loader, device, head="target"
        )

    result = {
        "seed": seed,
        "method": args.method,
        "selected_epoch": best["epoch"],
        "selected_checkpoint_score": best["checkpoint_score"],
        "source_val": final_source_val_metrics,
        "source_test": src_test_metrics,
        "target_dev": target_dev_metrics,
        "target_test": target_test_metrics,
        "selected_checkpoint_pseudo_diagnostics": best["pseudo_diagnostics"],
        "checkpoint_selection": {
            "criterion":
                "V7.4.4 label-free safe-checkpoint ensemble: hard anti-collapse-safe, signal-mature checkpoints only; average up to top-K checkpoints within an absolute score tolerance of the peak; fallback is never allowed to violate anti-collapse safety",
            "score_formula": {
                "source_val_macro_f1_effective_weight":
                    min(args.checkpoint_source_val_weight, args.checkpoint_source_val_weight_cap),
                "mean_selected_tri_score": args.checkpoint_consensus_weight,
                "signal_adaptation_maturity": args.checkpoint_maturity_weight,
                "selected_distribution_stability": args.v742_checkpoint_selected_stability_weight,
                "source_retention_trust": args.v742_checkpoint_source_retention_weight,
                "global_q_safety_soft": args.checkpoint_global_safety_weight,
                "mean_source_ema_js": -args.checkpoint_js_weight,
                "post_q_l1_drift": -args.checkpoint_drift_weight,
                "source_anchor_drift": -args.v742_checkpoint_anchor_drift_weight,
            },
            "soft_global_min_effective_classes_warning_threshold":
                args.checkpoint_min_global_effective_classes,
            "soft_global_max_hard_fraction_warning_threshold":
                args.checkpoint_max_global_hard_fraction,
            "min_selected_rate":
                args.checkpoint_min_selected_rate,
            "min_tri_consensus_rate":
                args.checkpoint_min_consensus_rate,
            "min_selected_effective_classes":
                args.checkpoint_min_selected_effective_classes,
            "max_selected_class_fraction":
                args.checkpoint_max_selected_class_fraction,
            "min_selected_tri_score":
                args.checkpoint_min_selected_tri_score,
            "max_source_ema_js":
                args.checkpoint_max_source_ema_js,
            "checkpoint_min_epoch": int(args.checkpoint_min_epoch),
            "late_tolerance_abs": float(args.checkpoint_late_tolerance_abs),
            "strict_peak_score_seen": float(strict_peak_score) if np.isfinite(strict_peak_score) else None,
            "fallback_peak_score_seen": float(fallback_best_score) if np.isfinite(fallback_best_score) else None,
            "fallback_used": bool(best.get("checkpoint_fallback_used", False)),
            "fallback_policy": best.get("checkpoint_fallback_policy"),
            "strict_gate_reasons_at_selected_epoch":
                best.get("strict_gate_reasons_at_selected_epoch"),
            "soft_gate_warnings_at_selected_epoch":
                best.get("soft_gate_warnings_at_selected_epoch"),
            "selected_controller_info": best.get("controller_info"),
            "selection_reference_source_val": best.get("source_val"),
            "checkpoint_ensemble_used": bool(best.get("checkpoint_ensemble_used", False)),
            "checkpoint_ensemble_epochs": best.get("checkpoint_ensemble_epochs", []),
            "checkpoint_ensemble_scores": best.get("checkpoint_ensemble_scores", []),
            "ensemble_top_k": int(args.ensemble_top_k),
            "ensemble_score_tolerance_abs": float(args.ensemble_score_tolerance_abs),
            "anti_collapse_rollback_count": int(anti_collapse_rollback_count),
        },
        "method_version": "OCRDA-v7.4.4",
        "architecture": {
            "visual_backbone": "ConvNeXt-Tiny",
            "classifier_heads": "dual source/target heads",
            "source_teacher": "frozen source-warm teacher",
            "target_teacher": "EMA target teacher",
            "pseudo_label_rule": "source==EMA==ontology prototype tri-consensus with tempered class quota",
            "pseudo_label_target": "ontology-smoothed soft semantic target; target prevalence prior decoupled from sampling prior",
            "prototype_geometry":
                "frozen source prototype + target EMA memory + ontology anchor",
            "covariance_alignment": False,
            "relation_aware_ontology_encoder": "R-GCN",
            "continuous_adaptation_gate": "selected-distribution-aware weighted geometric gate + hard anti-collapse rollback + causal plateau/drift controller",
            "checkpoint_ensemble": "label-free SWA-style average of anti-collapse-safe checkpoints",
        },
        "adaptation_gate": {
            "policy": "continuous selected-distribution-aware gate; global q(c) collapse is diagnostic only",
            "eff_low": args.adapt_gate_eff_low,
            "eff_high": args.adapt_gate_eff_high,
            "maxfrac_good": args.adapt_gate_maxfrac_good,
            "maxfrac_bad": args.adapt_gate_maxfrac_bad,
            "triscore_low": args.adapt_gate_triscore_low,
            "triscore_high": args.adapt_gate_triscore_high,
            "js_good": args.adapt_gate_js_good,
            "js_bad": args.adapt_gate_js_bad,
            "selected_rate_low": args.adapt_gate_selected_rate_low,
            "selected_rate_high": args.adapt_gate_selected_rate_high,
            "gate_floor": args.adapt_gate_floor,
        },
        "v742_plateau_drift_controller": {
            "policy": "causal next-epoch multiplier from actual rolling label-free signals",
            "signals": [
                "selected tri-score", "source/EMA JS",
                "selected-distribution TV drift", "prototype cosine drift",
                "source-anchor cosine drift", "source-retention drop"
            ],
            "maturity_threshold": args.v742_maturity_threshold,
            "plateau_patience": args.v742_plateau_patience,
            "plateau_decay": args.v742_plateau_decay,
            "plateau_floor": args.v742_plateau_floor,
            "controller_floor": args.v742_controller_floor,
            "global_q_collapse_hard_checkpoint_rejection": False,
        },
        "safeguards": {
            "target_adapt_labels_used": False,
            "target_dev_used_for_checkpoint_selection": False,
            "target_test_used_for_model_selection": False,
            "target_test_evaluated": bool(args.evaluate_final_test),
            "class_order": CLASS_NAMES,
            "anti_collapse_min_effective_classes": float(args.anti_collapse_min_effective_classes),
            "anti_collapse_max_class_fraction": float(args.anti_collapse_max_class_fraction),
            "anti_collapse_rollback_count": int(anti_collapse_rollback_count),
            "quota_temper_gamma": float(args.quota_temper_gamma),
            "quota_min_class_fraction": float(args.quota_min_class_fraction),
            "quota_min_class_count": int(args.quota_min_class_count),
        },
    }

    torch.save({
        "teacher_visual_state_dict": teacher_visual.state_dict(),
        "teacher_rgcn_state_dict": teacher_rgcn.state_dict(),
        "student_visual_state_dict": best["student_visual"],
        "student_rgcn_state_dict": best["student_rgcn"],
        "frozen_source_teacher_state_dict":
            source_teacher.state_dict() if source_teacher is not None else None,
        "frozen_source_rgcn_state_dict":
            source_teacher_rgcn.state_dict()
            if source_teacher_rgcn is not None else None,
        "source_prototypes": best.get("source_prototypes"),
        "prototype_memory_values": best.get("prototype_memory_values"),
        "prototype_memory_initialized": best.get("prototype_memory_initialized"),
        "prototype_memory_update_counts": best.get("prototype_memory_update_counts"),
        "ontology_nodes": [str(x) for x in ont["nodes"]],
        "ontology_relations": ont["rel_names"],
        "recognition_anchor_ids": ont["anchor_ids"].tolist(),
        "args": vars(args),
        "result": result,
    }, seed_dir / "best_model.pt")

    pd.DataFrame(history).to_csv(seed_dir / "history_v7_4_4.csv", index=False)
    pd.DataFrame(diagnostic_history).to_csv(
        seed_dir / "pseudo_diagnostics_history_v7_4_4.csv", index=False
    )
    (seed_dir / "metrics_v7_4_4.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    if best_snapshot is not None:
        save_pseudo_snapshot(
            best_snapshot,
            seed_dir / "target_adapt_tri_consensus_snapshot_best_v7_4_4.csv",
        )

    if target_dev_pred is not None:
        pred_df = pd.DataFrame({
            "sample_id": target_dev_pred["ids"],
            "y_true": target_dev_pred["y_true"],
            "y_pred": target_dev_pred["y_pred"],
        })
        for c, name in enumerate(CLASS_NAMES):
            pred_df[f"p_{name}"] = [
                x[c] for x in target_dev_pred["probs"]
            ]
        pred_df.to_csv(
            seed_dir / "target_dev_predictions_v7_4_4.csv", index=False
        )

    save_embeddings(
        teacher_visual, src_test_loader,
        seed_dir / "source_test_embeddings_v7_4_4.npz",
        device, head="source",
    )
    if tgt_dev_loader is not None:
        save_embeddings(
            teacher_visual, tgt_dev_loader,
            seed_dir / "target_dev_embeddings_v7_4_4.npz",
            device, head="target",
        )

    return result



def aggregate(results, outdir):
    def summarize(values):
        vals = [v for v in values if v is not None]
        if not vals:
            return None
        x = np.asarray(vals, dtype=float)
        mean = float(x.mean())
        sd = float(x.std(ddof=1)) if len(x) > 1 else None
        ci = (
            [mean - 1.96 * sd / math.sqrt(len(x)),
             mean + 1.96 * sd / math.sqrt(len(x))]
            if len(x) > 1 else None
        )
        return {
            "n": len(x), "mean": mean, "std": sd,
            "ci95_normal_approx": ci,
            "note": "CI is reported only for n>1 seeds." if len(x) == 1 else None,
        }

    obj = {
        "method_version": "OCRDA-v7.4.4",
        "n_seeds": len(results),
        "source_test_macro_f1":
            summarize([r["source_test"]["macro_f1"] for r in results]),
        "target_dev_macro_f1":
            summarize([
                r["target_dev"]["macro_f1"] if r.get("target_dev") else None
                for r in results
            ]),
        "target_test_macro_f1":
            summarize([
                r["target_test"]["macro_f1"] if r.get("target_test") else None
                for r in results
            ]),
        "per_seed": results,
    }
    (Path(outdir) / "aggregate_metrics_v7_4_4.json").write_text(
        json.dumps(obj, indent=2), encoding="utf-8"
    )
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--sdbd-root", required=True)
    ap.add_argument("--tsbd-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--method",
        choices=["full_ocrda"],
        default="full_ocrda",
    )
    ap.add_argument("--source-cache-dir", default=None)
    ap.add_argument("--use-source-cache", action="store_true")
    ap.add_argument("--max-adapt-steps-per-epoch", type=int, default=72)

    ap.add_argument("--source-train-file", default="source_train_final.csv")
    ap.add_argument("--source-val-file", default="source_val_final.csv")
    ap.add_argument("--source-test-file", default="source_test_final.csv")
    ap.add_argument("--target-adapt-file", default="target_adapt_final.csv")
    ap.add_argument("--target-dev-file", default=None)
    ap.add_argument("--target-test-file", default="target_test_final.csv")
    ap.add_argument("--report-target-dev", action="store_true")
    ap.add_argument("--evaluate-final-test", action="store_true")

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup-epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026])
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--rgcn-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--visual-dropout", type=float, default=0.20)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--backbone-lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--label-smoothing", type=float, default=0.05)

    # Dual-head source anchor.
    ap.add_argument("--lambda-target-source-anchor", type=float, default=0.20)
    ap.add_argument("--lambda-sem", type=float, default=0.15)
    ap.add_argument("--tau-sem", type=float, default=0.10)

    # Frozen source teacher + EMA target teacher.
    ap.add_argument("--teacher-ema", type=float, default=0.996)
    ap.add_argument("--lambda-soft-pseudo", type=float, default=0.60)
    ap.add_argument("--lambda-teacher-consistency", type=float, default=0.15)
    ap.add_argument("--lambda-source-teacher-consistency", type=float, default=0.20)
    ap.add_argument("--lambda-proto-contrast", type=float, default=0.20)
    ap.add_argument("--lambda-mean-alignment", type=float, default=0.10)
    ap.add_argument("--lambda-rank", type=float, default=0.15)

    # Tri-consensus.
    ap.add_argument("--source-conf-threshold", type=float, default=0.55)
    ap.add_argument("--ema-conf-threshold", type=float, default=0.55)
    ap.add_argument("--proto-conf-threshold", type=float, default=0.40)
    ap.add_argument("--source-ema-js-max", type=float, default=0.15)
    ap.add_argument("--consensus-score-min", type=float, default=0.15)
    ap.add_argument("--semantic-epsilon", type=float, default=0.06)
    ap.add_argument("--fusion-source-weight", type=float, default=0.35)
    ap.add_argument("--fusion-ema-weight", type=float, default=0.45)
    ap.add_argument("--fusion-proto-weight", type=float, default=0.20)

    # Source-anchored prototype geometry.
    ap.add_argument("--source-proto-weight", type=float, default=0.55)
    ap.add_argument("--memory-proto-weight", type=float, default=0.25)
    ap.add_argument("--ontology-proto-weight", type=float, default=0.20)
    ap.add_argument("--proto-vote-temperature", type=float, default=0.12)
    ap.add_argument("--proto-temperature", type=float, default=0.10)
    ap.add_argument("--prototype-memory-momentum", type=float, default=0.90)
    ap.add_argument("--prototype-epoch-momentum", type=float, default=0.85)
    ap.add_argument("--prototype-current-weight", type=float, default=0.20)
    ap.add_argument("--prototype-global-min-selected", type=int, default=1)

    # Conservative target pseudo-label curriculum.
    ap.add_argument("--pseudo-keep-ratio-start", type=float, default=0.10)
    ap.add_argument("--pseudo-keep-ratio-end", type=float, default=0.35)
    # v7.4.4: prevalence prior and sampling quota are deliberately decoupled.
    ap.add_argument("--quota-temper-gamma", type=float, default=0.50)
    ap.add_argument("--quota-min-class-fraction", type=float, default=0.10)
    ap.add_argument("--quota-min-class-count", type=int, default=6)

    # Target prior remains label-free, but source anchoring decays less
    # aggressively than v6.
    ap.add_argument("--prior-ema-momentum", type=float, default=0.65)
    ap.add_argument("--prior-source-weight-start", type=float, default=0.15)
    ap.add_argument("--prior-source-weight-end", type=float, default=0.02)
    ap.add_argument(
        "--prior-source-weight-schedule",
        choices=["cosine", "linear", "exponential"],
        default="cosine",
    )
    ap.add_argument("--prior-temperature", type=float, default=1.05)
    ap.add_argument("--prior-evidence-weight-consensus", type=float, default=0.35)
    ap.add_argument("--prior-evidence-weight-teacher-median", type=float, default=0.25)
    ap.add_argument("--prior-evidence-weight-fusion", type=float, default=0.25)
    ap.add_argument("--prior-evidence-weight-bbse", type=float, default=0.15)
    ap.add_argument("--prior-floor", type=float, default=0.02)
    ap.add_argument("--prior-ceiling", type=float, default=0.85)

    # Ontology topology. Covariance alignment is intentionally absent.
    ap.add_argument("--rank-margin", type=float, default=0.10)
    ap.add_argument("--rank-temperature", type=float, default=0.10)
    ap.add_argument("--topology-similarity-delta", type=float, default=0.05)

    ap.add_argument("--collapse-threshold", type=float, default=0.80)
    ap.add_argument(
        "--freeze-adaptation-on-collapse", action="store_true",
        help="Deprecated in v7.4.2. Global q(c) collapse is diagnostic only."
    )

    # OCRDA-v7.4.4 continuous adaptation gate, computed from selected
    # tri-consensus pseudo-labels rather than global hard q(c).
    ap.add_argument("--adapt-gate-eff-low", type=float, default=1.80)
    ap.add_argument("--adapt-gate-eff-high", type=float, default=2.70)
    ap.add_argument("--adapt-gate-maxfrac-good", type=float, default=0.50)
    ap.add_argument("--adapt-gate-maxfrac-bad", type=float, default=0.75)
    ap.add_argument("--adapt-gate-triscore-low", type=float, default=0.50)
    ap.add_argument("--adapt-gate-triscore-high", type=float, default=0.80)
    ap.add_argument("--adapt-gate-js-good", type=float, default=0.02)
    ap.add_argument("--adapt-gate-js-bad", type=float, default=0.15)
    ap.add_argument("--adapt-gate-selected-rate-low", type=float, default=0.03)
    ap.add_argument("--adapt-gate-selected-rate-high", type=float, default=0.12)
    ap.add_argument("--adapt-gate-ready-selected-rate", type=float, default=0.01)
    ap.add_argument("--adapt-gate-ready-consensus-rate", type=float, default=0.05)
    ap.add_argument("--adapt-gate-floor", type=float, default=0.10)
    ap.add_argument("--adapt-gate-epsilon", type=float, default=1e-4)
    ap.add_argument("--adapt-gate-weight-eff", type=float, default=0.25)
    ap.add_argument("--adapt-gate-weight-balance", type=float, default=0.25)
    ap.add_argument("--adapt-gate-weight-tri", type=float, default=0.20)
    ap.add_argument("--adapt-gate-weight-js", type=float, default=0.15)
    ap.add_argument("--adapt-gate-weight-coverage", type=float, default=0.15)


    # OCRDA-v7.4.4: class-balanced semantic adaptation.
    ap.add_argument(
        "--class-balance-gamma",
        type=float,
        default=0.75,
    )
    ap.add_argument(
        "--class-balance-min",
        type=float,
        default=0.60,
    )
    ap.add_argument(
        "--class-balance-max",
        type=float,
        default=1.80,
    )

    # OCRDA-v7.4.4: dual-distribution safety.
    ap.add_argument(
        "--dual-gate-selected-mix",
        type=float,
        default=0.75,
    )
    ap.add_argument(
        "--dual-gate-global-eff-low",
        type=float,
        default=1.50,
    )
    ap.add_argument(
        "--dual-gate-global-eff-high",
        type=float,
        default=2.50,
    )
    ap.add_argument(
        "--dual-gate-global-maxfrac-good",
        type=float,
        default=0.55,
    )
    ap.add_argument(
        "--dual-gate-global-maxfrac-bad",
        type=float,
        default=0.88,
    )

    # OCRDA-v7.4.4: adaptation maturity.
    ap.add_argument(
        "--maturity-active-ramp-threshold",
        type=float,
        default=0.02,
    )
    ap.add_argument(
        "--checkpoint-min-effective-ramp",
        type=float,
        default=0.25,
    )
    ap.add_argument(
        "--checkpoint-min-active-adaptation-epochs",
        type=int,
        default=4,
    )
    ap.add_argument(
        "--checkpoint-min-cumulative-adaptation-mass",
        type=float,
        default=0.20,
    )

    # Soft global checkpoint safety.
    ap.add_argument(
        "--checkpoint-min-global-effective-classes",
        type=float,
        default=1.65,
    )
    ap.add_argument(
        "--checkpoint-max-global-hard-fraction",
        type=float,
        default=0.83,
    )

    # Reduce early source-validation dominance.
    ap.add_argument(
        "--checkpoint-source-val-weight-cap",
        type=float,
        default=0.25,
    )
    ap.add_argument(
        "--checkpoint-maturity-weight",
        type=float,
        default=0.18,
    )
    ap.add_argument(
        "--checkpoint-global-safety-weight",
        type=float,
        default=0.02,
    )
    ap.add_argument(
        "--checkpoint-maturity-ramp-reference",
        type=float,
        default=0.50,
    )
    ap.add_argument(
        "--checkpoint-maturity-mass-reference",
        type=float,
        default=2.00,
    )



    # OCRDA-v7.4.4: actual rolling plateau/drift controller.
    ap.add_argument("--v742-min-history-epochs", type=int, default=3)
    ap.add_argument("--v742-tri-score-low", type=float, default=0.72)
    ap.add_argument("--v742-tri-score-high", type=float, default=0.84)
    ap.add_argument("--v742-js-good", type=float, default=0.025)
    ap.add_argument("--v742-js-bad", type=float, default=0.10)
    ap.add_argument("--v742-selected-drift-good", type=float, default=0.025)
    ap.add_argument("--v742-selected-drift-bad", type=float, default=0.16)
    ap.add_argument("--v742-prototype-drift-good", type=float, default=0.008)
    ap.add_argument("--v742-prototype-drift-bad", type=float, default=0.08)
    ap.add_argument("--v742-source-anchor-drift-good", type=float, default=0.02)
    ap.add_argument("--v742-source-anchor-drift-bad", type=float, default=0.12)

    ap.add_argument("--v742-maturity-weight-tri", type=float, default=0.30)
    ap.add_argument("--v742-maturity-weight-js", type=float, default=0.15)
    ap.add_argument("--v742-maturity-weight-selected-stability", type=float, default=0.25)
    ap.add_argument("--v742-maturity-weight-prototype-stability", type=float, default=0.15)
    ap.add_argument("--v742-maturity-weight-anchor-stability", type=float, default=0.15)
    ap.add_argument("--v742-maturity-threshold", type=float, default=0.70)
    ap.add_argument("--v742-checkpoint-min-maturity", type=float, default=0.62)
    ap.add_argument("--v742-plateau-patience", type=int, default=5)
    ap.add_argument("--v742-plateau-decay", type=float, default=0.97)
    ap.add_argument("--v742-plateau-floor", type=float, default=0.65)
    ap.add_argument("--v742-controller-floor", type=float, default=0.35)

    ap.add_argument("--v742-source-retention-drop-good", type=float, default=0.005)
    ap.add_argument("--v742-source-retention-drop-bad", type=float, default=0.05)
    ap.add_argument("--v742-source-retention-floor", type=float, default=0.40)
    ap.add_argument("--v742-drift-alarm-stability", type=float, default=0.30)
    ap.add_argument("--v742-drift-guard-floor", type=float, default=0.45)

    ap.add_argument("--v742-checkpoint-selected-stability-weight", type=float, default=0.15)
    ap.add_argument("--v742-checkpoint-source-retention-weight", type=float, default=0.10)
    ap.add_argument("--v742-checkpoint-anchor-drift-weight", type=float, default=0.03)
    ap.add_argument("--v742-fallback-min-active-epochs", type=int, default=2)

    # v7.4.4 hard anti-collapse rollback.
    ap.add_argument("--anti-collapse-min-effective-classes", type=float, default=2.00)
    ap.add_argument("--anti-collapse-max-class-fraction", type=float, default=0.75)
    ap.add_argument("--anti-collapse-rollback-multiplier", type=float, default=0.25)

    # v7.4.4 safe-checkpoint SWA-style ensemble.
    ap.add_argument("--ensemble-top-k", type=int, default=5)
    ap.add_argument("--ensemble-min-count", type=int, default=2)
    ap.add_argument("--ensemble-max-pool", type=int, default=8)
    ap.add_argument("--ensemble-pool-tolerance-abs", type=float, default=0.05)
    ap.add_argument("--ensemble-score-tolerance-abs", type=float, default=0.03)

    # Label-free checkpoint safety and independent-consensus score.
    ap.add_argument("--checkpoint-min-effective-classes", type=float, default=2.30)
    ap.add_argument("--checkpoint-max-pseudo-fraction", type=float, default=0.70)
    ap.add_argument("--checkpoint-min-epoch", type=int, default=16)
    ap.add_argument("--checkpoint-min-selected-rate", type=float, default=0.03)
    ap.add_argument("--checkpoint-min-consensus-rate", type=float, default=0.05)

    # v7.1: safety is evaluated on selected tri-consensus candidates.
    ap.add_argument(
        "--checkpoint-min-selected-effective-classes",
        type=float, default=2.00
    )
    ap.add_argument(
        "--checkpoint-max-selected-class-fraction",
        type=float, default=0.70
    )
    ap.add_argument(
        "--checkpoint-min-selected-tri-score",
        type=float, default=0.50
    )
    ap.add_argument(
        "--checkpoint-max-source-ema-js",
        type=float, default=0.15
    )

    ap.add_argument("--checkpoint-source-val-weight", type=float, default=0.25)
    ap.add_argument("--checkpoint-consensus-weight", type=float, default=0.30)
    ap.add_argument("--checkpoint-js-weight", type=float, default=0.03)
    ap.add_argument("--checkpoint-drift-weight", type=float, default=0.02)
    ap.add_argument("--checkpoint-late-tolerance-abs", type=float, default=0.015,
                    help="Choose the latest eligible checkpoint within this absolute score tolerance of the best label-free score.")

    args = ap.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    ont = ontology_semantic_graph(args.ontology)
    summary = {
        "version": "OCRDA-v7.4.4",
        "nodes": len(ont["nodes"]),
        "directed_edges_including_reverse": int(len(ont["src"])),
        "relation_types_including_reverse": len(ont["rel_names"]),
        "relations": ont["rel_names"],
        "class_order": CLASS_NAMES,
        "recognition_anchors": [ANCHOR_CLASS_LOCAL[c] for c in CLASS_NAMES],
        "anchor_specificity": ont["specificity"].tolist(),
        "ontology_similarity": ont["topology"].tolist(),
        "recognition_topology": ont["topology"].tolist(),
        "context_topology_diagnostic": ont["context_topology"].tolist(),
        "recognition_topology_relations": ont["recognition_topology_relations"],
        "recognition_topology_tau": RECOGNITION_TOPOLOGY_TAU,
        "recognition_topology_max_offdiag": RECOGNITION_TOPOLOGY_MAX_OFFDIAG,
        "architecture": {
            "visual_backbone": "ConvNeXt-Tiny",
            "dual_classifier_heads": True,
            "frozen_source_teacher": True,
            "ema_target_teacher": True,
            "ontology_tri_consensus": True,
            "source_anchored_prototypes": True,
            "soft_semantic_pseudo_labels": True,
            "covariance_alignment": False,
            "relation_aware_ontology_encoder": "R-GCN",
            "continuous_adaptation_gate": "selected-distribution-aware weighted geometric gate + causal plateau/drift controller",
        },
        "v742_controller_core": "rolling selected-distribution/prototype/source-anchor drift + source-retention trust",
        "v744_changes": {
            "recognition_only_topology": True,
            "context_relations_excluded_from_label_smoothing": ["COOCCURSWITH", "HASPATHOGEN"],
            "robust_multievidence_target_prior": True,
            "bbse_uses_source_validation_only": True,
            "strict_min_checkpoint_epoch": 16,
            "late_stable_checkpoint_selection": True,
        },
        "ontology_graph_leakage_safeguard": {
            "alignmentEligible_false_edges_excluded": True,
            "dataset_species_learningDomain_provenance_edges_unavailable_to_encoder": True,
        },
    }
    (Path(args.output_dir) / "ontology_graph_summary_v7_4_4.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

    results = [train_seed(args, s, ont) for s in args.seeds]
    agg = aggregate(results, args.output_dir)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
