"""Tests for GRU with temporal attention model."""

import pytest
import torch

from src.models.gru import GRUWithAttention, TemporalAttention, count_parameters


class TestTemporalAttention:
    def test_output_shapes(self):
        attn = TemporalAttention(hidden_size=32, attention_size=16)
        h = torch.randn(4, 10, 32)
        context, weights = attn(h)
        assert context.shape == (4, 32)
        assert weights.shape == (4, 10)

    def test_weights_sum_to_one(self):
        attn = TemporalAttention(hidden_size=32)
        h = torch.randn(4, 10, 32)
        _, weights = attn(h)
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(4), atol=1e-5, rtol=0)

    def test_padding_mask_zeros_padded(self):
        attn = TemporalAttention(hidden_size=16)
        h = torch.randn(2, 5, 16)
        mask = torch.tensor([
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
        ], dtype=torch.float32)

        _, weights = attn(h, padding_mask=mask)

        # Padded positions must have exactly 0 weight
        assert weights[0, 0].item() == 0.0
        assert weights[0, 1].item() == 0.0
        assert weights[1, 0].item() == 0.0
        assert weights[1, 1].item() == 0.0
        assert weights[1, 2].item() == 0.0

        # Non-padded weights must be positive and sum to 1
        assert (weights[0, 2:] > 0).all()
        assert (weights[1, 3:] > 0).all()
        assert weights[0, 2:].sum().item() == pytest.approx(1.0, abs=1e-5)
        assert weights[1, 3:].sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_all_padded_no_nan(self):
        attn = TemporalAttention(hidden_size=16)
        h = torch.randn(1, 3, 16)
        mask = torch.zeros(1, 3)
        context, weights = attn(h, padding_mask=mask)
        assert not torch.isnan(weights).any()
        assert not torch.isnan(context).any()


class TestGRUWithAttention:
    def test_forward_shapes(self):
        model = GRUWithAttention(input_size=120, hidden_size=128, bidirectional=True)
        x = torch.randn(8, 24, 120)
        preds, weights = model(x)
        assert preds.shape == (8,)
        assert weights.shape == (8, 24)

    def test_output_range(self):
        model = GRUWithAttention(input_size=20, hidden_size=32, num_layers=1)
        model.eval()
        x = torch.randn(16, 10, 20)
        with torch.no_grad():
            probs, _ = model.predict_proba(x)
        # Sigmoid output must be in (0, 1)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_padding_mask_propagates(self):
        model = GRUWithAttention(
            input_size=4, hidden_size=16, num_layers=1, bidirectional=False, dropout=0.0,
        )
        model.eval()
        x = torch.randn(2, 6, 4)
        mask = torch.tensor([
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1],
        ], dtype=torch.float32)

        with torch.no_grad():
            preds, weights = model(x, padding_mask=mask)

        # Padded positions should have 0 attention
        assert (weights[0, :3] == 0).all()
        assert (weights[1, :4] == 0).all()
        # Non-padded should sum to 1
        assert weights[0, 3:].sum().item() == pytest.approx(1.0, abs=1e-5)
        assert weights[1, 4:].sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_bidirectional_true(self):
        model = GRUWithAttention(input_size=8, hidden_size=16, bidirectional=True)
        x = torch.randn(4, 5, 8)
        preds, weights = model(x)
        assert preds.shape == (4,)
        assert weights.shape == (4, 5)

    def test_bidirectional_false(self):
        model = GRUWithAttention(input_size=8, hidden_size=16, bidirectional=False)
        x = torch.randn(4, 5, 8)
        preds, weights = model(x)
        assert preds.shape == (4,)
        assert weights.shape == (4, 5)

    def test_bidirectional_has_more_params(self):
        model_bi = GRUWithAttention(input_size=8, hidden_size=16, bidirectional=True)
        model_uni = GRUWithAttention(input_size=8, hidden_size=16, bidirectional=False)
        total_bi, _ = count_parameters(model_bi)
        total_uni, _ = count_parameters(model_uni)
        assert total_bi > total_uni

    def test_single_sample_batch(self):
        model = GRUWithAttention(input_size=10, hidden_size=16, num_layers=1)
        x = torch.randn(1, 5, 10)
        preds, weights = model(x)
        assert preds.shape == (1,)


class TestCountParameters:
    def test_all_trainable(self):
        model = GRUWithAttention(input_size=120, hidden_size=128, num_layers=2, bidirectional=True)
        total, trainable = count_parameters(model)
        assert total == trainable
        assert total > 0

    def test_print_count(self):
        model = GRUWithAttention(input_size=120, hidden_size=128, num_layers=2, bidirectional=True)
        total, trainable = count_parameters(model)
        print(f"\nFull model: {total:,} total params, {trainable:,} trainable")

        model_uni = GRUWithAttention(
            input_size=120, hidden_size=128, num_layers=2, bidirectional=False,
        )
        total_uni, _ = count_parameters(model_uni)
        print(f"Unidirectional: {total_uni:,} total params")
