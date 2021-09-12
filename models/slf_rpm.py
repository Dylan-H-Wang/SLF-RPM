import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet3d import generate_model

LARGE_NUM = 1e9


class SLF_RPM(nn.Module):
    """Model architeture for SLF-RPM."""

    def __init__(
        self,
        model_depth: str,
        n_class: int,
        temperature: float,
        n_spatial: int,
        n_temporal: int,
    ) -> None:
        """
        Args:
            model_depth (str): Depth of backbone model.
            n_class (int): Number of prediction classes.
            temperature (float): Hyperparameter for `tau`
            n_spatial (int): Number of ROIs for spatial augmentation.
            n_temporal (int): Number of strides for temporal augmentation.
        """
        super(SLF_RPM, self).__init__()

        self.temperature = temperature

        # Backbone
        self.encoder_q = generate_model(model_depth=model_depth, n_classes=n_class)

        # Projection head
        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.proj_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, n_class)
        )
        self.encoder_q.fc = nn.Identity()

        # Augmentation classifier
        self.spatial_classifer = nn.Linear(dim_mlp, n_spatial)
        self.temporal_classifer = nn.Linear(dim_mlp, n_temporal)

    def forward(self, vids: torch.Tensor):
        """
        Args:
            vids (Tensor): Inputs with shape of (2*n_video, n_frame, n_channel, height, width).

        Returns:
            A tuple containing:
                Logits of current mini-batch.

                Indicator for positive/negative sameples.

                A list of predicted spatial augmentations.

                A list of predicted temporal augmentations.
        """
        x = torch.cat(vids, axis=0).transpose(
            1, 2
        )  # (2*n_video, n_channel, n_frame, height, width)

        # Compute video features
        q = self.encoder_q(x)  # (2*n_video, features)

        # Predict augmentation class
        pred_spatial = self.spatial_classifer(q)
        pred_temporal = self.temporal_classifer(q)

        # Compute similarity
        q = self.proj_head(q)  # # (2*n_video, proj_features)
        q = F.normalize(q, dim=-1)

        feature_a = q[0 : q.shape[0] // 2]  # (n_video, features)
        feature_b = q[q.shape[0] // 2 : q.shape[0]]  # (n_video, features)

        # Compute positive and negative logits
        batch_size = feature_a.shape[0]
        masks = F.one_hot(
            torch.arange(0, batch_size, device=feature_a.device), num_classes=batch_size
        )  # (n_video, features)

        logits_aa = torch.matmul(feature_a, feature_a.T)  # (n_video, n_video)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(feature_b, feature_b.T)  # (n_video, n_video)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(feature_a, feature_b.T)  # (n_video, n_video)
        logits_ba = torch.matmul(feature_b, feature_a.T)  # (n_video, n_video)

        logits_a = torch.cat([logits_ab, logits_aa], dim=1)  # (n_video, 2*n_video)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)  # (n_video, 2*n_video)
        logits = (
            torch.cat([logits_a, logits_b], axis=0) / self.temperature
        )  # (2*n_video, 2*n_video)

        labels = torch.arange(0, batch_size, device=feature_a.device)  # (n_video,)
        labels = torch.cat([labels, labels], axis=0)  # (2*n_video,)

        return logits, labels, pred_spatial, pred_temporal
