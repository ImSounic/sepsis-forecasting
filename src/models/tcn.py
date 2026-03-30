"""Temporal Convolutional Network with attention for sepsis prediction.

Causal by design — each timestep can only see the present and past,
never the future. This avoids the data leakage problem of bidirectional RNNs
in real-time clinical prediction tasks.

Architecture: LayerNorm -> Causal TCN -> Temporal Attention -> FC layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gru import TemporalAttention


class CausalConv1dBlock(nn.Module):
    """Residual block with two causal dilated convolutions.

    Causality is enforced by left-padding the input before each convolution
    so that the output at time t depends only on inputs at times <= t.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        self.causal_pad1 = (kernel_size - 1) * dilation
        self.causal_pad2 = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               dilation=dilation, padding=0)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               dilation=dilation, padding=0)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # 1x1 conv for residual when channel dimensions differ
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, out_channels, seq_len) — same temporal dimension.
        """
        residual = self.residual(x)

        out = F.pad(x, (self.causal_pad1, 0))
        out = self.dropout(torch.relu(self.bn1(self.conv1(out))))

        out = F.pad(out, (self.causal_pad2, 0))
        out = self.dropout(torch.relu(self.bn2(self.conv2(out))))

        return torch.relu(out + residual)


class TemporalConvNet(nn.Module):
    """Stack of causal residual blocks with exponentially increasing dilation.

    Dilations [1, 2, 4, 8] with kernel_size=3 give a receptive field of 61,
    covering the full 24-timestep input window.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            dilation = 2 ** i
            layers.append(CausalConv1dBlock(in_ch, hidden_size, kernel_size,
                                            dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, hidden_size, seq_len)
        """
        return self.network(x)


class TCNWithAttention(nn.Module):
    """Causal TCN with temporal attention for binary classification.

    Architecture: LayerNorm -> TCN -> Attention -> FC(hidden,64) -> ReLU -> FC(64,1)

    Matches the GRUWithAttention interface exactly:
    - forward(x, padding_mask) -> (logits, attention_weights)
    - predict_proba(x, padding_mask) -> (probabilities, attention_weights)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        kernel_size: int = 3,
        attention_size: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.layer_norm = nn.LayerNorm(input_size)

        self.tcn = TemporalConvNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.attention = TemporalAttention(hidden_size, attention_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: input features, (batch, seq_len, input_size).
            padding_mask: (batch, seq_len), 1 = real, 0 = padded.

        Returns:
            logits: raw scores before sigmoid, (batch,).
            attention_weights: per-timestep weights, (batch, seq_len).
        """
        x = self.layer_norm(x)

        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        # Back to (batch, seq_len, hidden_size)
        x = x.transpose(1, 2)

        # Zero out padded positions before attention
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1)

        context, attn_weights = self.attention(x, padding_mask)

        out = self.dropout(context)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out).squeeze(-1)  # (batch,)

        return out, attn_weights

    def predict_proba(self, x, padding_mask=None):
        """Forward pass returning sigmoid probabilities for inference.

        Returns:
            probabilities: values in (0, 1), (batch,).
            attention_weights: per-timestep weights, (batch, seq_len).
        """
        logits, attn_weights = self.forward(x, padding_mask)
        return torch.sigmoid(logits), attn_weights
