import torch.nn as nn

from .resnet3d import generate_model


class LinearClsResNet3D(nn.Module):
    """ Classifier model for contrastive learning.
    """

    def __init__(
        self,
        model_depth: int,
        n_class: int,
        dropout: float = 0.0,
        norm_layer=nn.BatchNorm3d,
    ) -> None:
        """
        Args:
            model_depth (int): Depth of classifier model.
            n_class (int): Number of prediction classes.
            dropout (float, optional): Dropout rate for the classifier model. Defaults to 0.0.
            norm_layer (optional): Normalisation layer for classifier model. Defaults to `nn.BatchNorm3d`.
        """
        super(LinearClsResNet3D, self).__init__()

        self.encoder_q = generate_model(
            model_depth=model_depth,
            n_classes=n_class,
            dropout=dropout,
            norm_layer=norm_layer,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input with shape of (n_video, n_frame, n_channel, height, width).

        Returns:
            Tensor: Results with shape of (n_video, n_class).
        """

        x = x.transpose(1, 2)  # (n_video, n_channel, n_frame, height, width)
        preds = self.encoder_q(x)  # (n_video, n_class)
        return preds
