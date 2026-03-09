"""GRU with temporal attention for sepsis prediction."""

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention over time steps.

    Learns a scoring function: score_t = v^T * tanh(W * h_t + b)
    then normalizes via softmax to produce attention weights.
    """

    def __init__(self, hidden_size: int, attention_size: int = 64):
        super().__init__()
        self.project = nn.Linear(hidden_size, attention_size)
        self.score = nn.Linear(attention_size, 1, bias=False)

    def forward(self, h, padding_mask=None):
        """
        Args:
            h: GRU outputs, (batch, seq_len, hidden_size).
            padding_mask: (batch, seq_len), 1 = real, 0 = padded.

        Returns:
            context: weighted sum, (batch, hidden_size).
            weights: attention distribution, (batch, seq_len).
        """
        energy = torch.tanh(self.project(h))       # (batch, seq_len, attention_size)
        scores = self.score(energy).squeeze(-1)     # (batch, seq_len)

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        # All-padded rows produce NaN from softmax(-inf); replace with zeros
        weights = torch.nan_to_num(weights, nan=0.0)

        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (batch, hidden_size)
        return context, weights


class GRUWithAttention(nn.Module):
    """Multi-layer GRU with temporal attention for binary classification.

    Architecture: LayerNorm -> GRU -> Attention -> FC(hidden,64) -> ReLU -> FC(64,1)

    forward() returns raw logits (for use with BCEWithLogitsLoss).
    Use predict_proba() to get sigmoid probabilities for inference.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        attention_size: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        effective_hidden = hidden_size * (2 if bidirectional else 1)

        self.layer_norm = nn.LayerNorm(input_size)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.attention = TemporalAttention(effective_hidden, attention_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(effective_hidden, 64)
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

        gru_out, _ = self.gru(x)  # (batch, seq_len, effective_hidden)

        context, attn_weights = self.attention(gru_out, padding_mask)

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


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params) for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
