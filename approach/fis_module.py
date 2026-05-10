import random
import torch
import torch.nn.functional as F


class FISGenerator:

    def __init__(self,
                 device='cuda',
                 lambda_min=0.2,
                 lambda_max=0.8,
                 topk_ratio=0.3,
                 severe_class_id=3,
                 filter_threshold=0.5,
                 manifold_threshold=1.0):

        self.device = device

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        self.topk_ratio = topk_ratio

        self.severe_class_id = severe_class_id

        self.filter_threshold = filter_threshold
        self.manifold_threshold = manifold_threshold

    # ======================================================
    # Gradient-weighted feature importance
    # ======================================================
    def compute_importance(self,
                           features,
                           gradients):

        """
        features: (B, D)
        gradients: (B, D)

        importance = |grad * feature|
        """

        importance = torch.abs(features * gradients)

        return importance

    # ======================================================
    # Select discriminative subspace
    # ======================================================
    def select_topk_dimensions(self,
                               importance):

        D = importance.size(1)

        k = int(D * self.topk_ratio)
        k = max(1, k)

        mean_importance = importance.mean(dim=0)

        topk_idx = torch.topk(mean_importance, k).indices

        return topk_idx

    # ======================================================
    # Semantic alignment
    # ======================================================
    def semantic_alignment(self,
                           feat_a,
                           feat_b):

        """
        cosine similarity alignment
        """

        sim = F.cosine_similarity(
            feat_a.unsqueeze(0),
            feat_b.unsqueeze(0)
        ).item()

        return sim

    # ======================================================
    # Importance-aware interpolation
    # ======================================================
    def interpolate(self,
                    feat_minor,
                    feat_major,
                    important_dims):

        lambda_value = random.uniform(
            self.lambda_min,
            self.lambda_max
        )

        synthetic = feat_minor.clone()

        synthetic[important_dims] = (
            feat_minor[important_dims]
            + lambda_value * (
                feat_major[important_dims]
                - feat_minor[important_dims]
            )
        )

        return synthetic

    # ======================================================
    # Confidence filter
    # ======================================================
    def confidence_filter(self,
                          model,
                          synthetic_feature,
                          severe_head):

        with torch.no_grad():

            pred = severe_head(
                synthetic_feature.unsqueeze(0)
            )

            confidence = torch.sigmoid(pred).mean().item()

        return confidence >= self.filter_threshold

    # ======================================================
    # Manifold distance filter
    # ======================================================
    def manifold_filter(self,
                        synthetic_feature,
                        severe_prototype):

        dist = torch.norm(
            synthetic_feature - severe_prototype,
            p=2
        ).item()

        return dist <= self.manifold_threshold

    # ======================================================
    # Main synthesis function
    # ======================================================
    def generate(self,
                 severe_features,
                 majority_features,
                 importance_scores,
                 severe_prototype,
                 model=None,
                 severe_head=None,
                 num_generate=20):

        synthetic_samples = []

        important_dims = self.select_topk_dimensions(
            importance_scores
        )

        for _ in range(num_generate):

            severe_idx = random.randint(
                0,
                severe_features.size(0) - 1
            )

            majority_idx = random.randint(
                0,
                majority_features.size(0) - 1
            )

            feat_minor = severe_features[severe_idx]
            feat_major = majority_features[majority_idx]

            # ==========================================
            # semantic alignment
            # ==========================================
            sim = self.semantic_alignment(
                feat_minor,
                feat_major
            )

            if sim < 0.3:
                continue

            # ==========================================
            # interpolation
            # ==========================================
            synthetic = self.interpolate(
                feat_minor,
                feat_major,
                important_dims
            )

            # ==========================================
            # manifold filter
            # ==========================================
            if not self.manifold_filter(
                    synthetic,
                    severe_prototype):
                continue

            # ==========================================
            # confidence filter
            # ==========================================
            if model is not None and severe_head is not None:
                if not self.confidence_filter(
                        model,
                        synthetic,
                        severe_head):
                    continue

            synthetic_samples.append(synthetic)

        if len(synthetic_samples) == 0:
            return None

        synthetic_samples = torch.stack(
            synthetic_samples,
            dim=0
        )

        return synthetic_samples