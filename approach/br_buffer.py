import math
import random
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class BRDataset(Dataset):
    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item["image"], item["target"]


class BalancedReplayBuffer:
    """
    Balanced Replay memory buffer.

    Each stored sample:
    {
        "image": Tensor(C,T,H,W) or Tensor(C,H,W),
        "target": Tensor scalar,
        "feature": Tensor(D),
        "attn_score": float,
        "uncertainty": float,
        "class_id": int
    }
    """

    def __init__(
        self,
        memory_size: int = 200,
        proto_ratio: float = 0.8,
        uncertainty_ratio: float = 0.2,
        device: str = "cuda",
    ):
        self.memory_size = memory_size
        self.proto_ratio = proto_ratio
        self.uncertainty_ratio = uncertainty_ratio
        self.device = device
        self.samples: List[dict] = []

    def __len__(self):
        return len(self.samples)

    def is_empty(self):
        return len(self.samples) == 0

    def get_dataset(self):
        return BRDataset(self.samples)

    def get_all_samples(self):
        return self.samples

    @staticmethod
    def _to_class_id(target):
        """
        If your target is already y=0,1,2,3, this directly works.
        If your target is BDI score, it maps BDI-II score to 4 severity levels.
        """
        if torch.is_tensor(target):
            value = float(target.detach().cpu().view(-1)[0])
        else:
            value = float(target)

        # If target is already class label
        if value in [0, 1, 2, 3]:
            return int(value)

        # BDI-II mapping
        if value <= 13:
            return 0
        elif value <= 19:
            return 1
        elif value <= 28:
            return 2
        else:
            return 3

    @staticmethod
    def _safe_float(x):
        if torch.is_tensor(x):
            return float(x.detach().cpu().view(-1)[0])
        return float(x)

    def update(self, new_candidates: List[dict]):
        """
        Update memory using:
        1) class balance
        2) attention-guided candidate filtering
        3) prototype samples, 80%
        4) uncertainty samples, 20%
        """
        if len(new_candidates) == 0:
            return

        all_candidates = self.samples + new_candidates

        grouped: Dict[int, List[dict]] = {}
        for item in all_candidates:
            class_id = int(item["class_id"])
            grouped.setdefault(class_id, []).append(item)

        num_classes = max(len(grouped), 1)
        per_class_budget = max(1, self.memory_size // num_classes)

        selected = []
        for class_id, items in grouped.items():
            if len(items) <= per_class_budget:
                selected.extend(items)
                continue

            # Step 1: attention-guided pre-filter.
            # Keep more candidates than final budget to avoid over-filtering.
            items = sorted(items, key=lambda z: z["attn_score"], reverse=True)
            keep_num = min(len(items), max(per_class_budget * 2, per_class_budget))
            items = items[:keep_num]

            feats = torch.stack([z["feature"].float() for z in items], dim=0)
            proto = feats.mean(dim=0, keepdim=True)
            dists = torch.norm(feats - proto, p=2, dim=1)

            proto_num = int(round(per_class_budget * self.proto_ratio))
            proto_num = max(1, min(proto_num, per_class_budget))

            uncertainty_num = per_class_budget - proto_num

            # Prototype samples: closest to class center
            proto_indices = torch.argsort(dists)[:proto_num].tolist()

            chosen_idx = set(proto_indices)

            # Uncertainty samples: highest uncertainty among remaining
            remaining = [i for i in range(len(items)) if i not in chosen_idx]
            if uncertainty_num > 0 and len(remaining) > 0:
                remaining_sorted = sorted(
                    remaining,
                    key=lambda i: items[i]["uncertainty"],
                    reverse=True,
                )
                uncertainty_indices = remaining_sorted[:uncertainty_num]
                chosen_idx.update(uncertainty_indices)

            selected.extend([items[i] for i in sorted(chosen_idx)])

        if len(selected) > self.memory_size:
            selected = sorted(selected, key=lambda z: z["attn_score"], reverse=True)
            selected = selected[:self.memory_size]

        random.shuffle(selected)
        self.samples = selected