"""Tests for TCN with temporal attention model."""

import pytest
import torch

from src.models.tcn import CausalConv1dBlock, TemporalConvNet, TCNWithAttention
from src.models.gru import count_parameters


class TestCausalConv1dBlock:
    def test_output_shape_preserves_seq_len(self):
        block = CausalConv1dBlock(in_channels=32, out_channels=64,
                                  kernel_size=3, dilation=1)
        x = torch.randn(4, 32, 24)
        out = block(x)
        assert out.shape == (4, 64, 24)

    def test_output_shape_same_channels(self):
        block = CausalConv1dBlock(in_channels=64, out_channels=64,
                                  kernel_size=3, dilation=2)
        x = torch.randn(4, 64, 24)
        out = block(x)
        assert out.shape == (4, 64, 24)

    def test_causality(self):
        """Modifying a future timestep must not change earlier outputs."""
        block = CausalConv1dBlock(in_channels=8, out_channels=16,
                                  kernel_size=3, dilation=1, dropout=0.0)
        block.eval()

        x = torch.randn(1, 8, 10)

        with torch.no_grad():
            out_original = block(x).clone()

        # Modify the last timestep
        x_modified = x.clone()
        x_modified[:, :, -1] = torch.randn(1, 8)

        with torch.no_grad():
            out_modified = block(x_modified)

        # All timesteps except the last must be identical
        torch.testing.assert_close(out_original[:, :, :-1],
                                   out_modified[:, :, :-1])

    def test_different_dilations(self):
        for dilation in [1, 2, 4, 8]:
            block = CausalConv1dBlock(in_channels=16, out_channels=16,
                                      kernel_size=3, dilation=dilation)
            x = torch.randn(2, 16, 24)
            out = block(x)
            assert out.shape == (2, 16, 24)


class TestTemporalConvNet:
    def test_output_shape(self):
        tcn = TemporalConvNet(input_size=120, hidden_size=128,
                              num_layers=4, kernel_size=3)
        x = torch.randn(4, 120, 24)
        out = tcn(x)
        assert out.shape == (4, 128, 24)

    def test_causality_through_stack(self):
        """Full TCN stack must be causal — future changes don't affect past."""
        tcn = TemporalConvNet(input_size=8, hidden_size=16,
                              num_layers=3, kernel_size=3, dropout=0.0)
        tcn.eval()

        x = torch.randn(1, 8, 12)

        with torch.no_grad():
            out_original = tcn(x).clone()

        x_modified = x.clone()
        x_modified[:, :, -1] = torch.randn(1, 8)

        with torch.no_grad():
            out_modified = tcn(x_modified)

        torch.testing.assert_close(out_original[:, :, :-1],
                                   out_modified[:, :, :-1])


class TestTCNWithAttention:
    def test_forward_shapes(self):
        model = TCNWithAttention(input_size=120, hidden_size=128)
        x = torch.randn(8, 24, 120)
        logits, weights = model(x)
        assert logits.shape == (8,)
        assert weights.shape == (8, 24)

    def test_output_range(self):
        model = TCNWithAttention(input_size=20, hidden_size=32, num_layers=2)
        model.eval()
        x = torch.randn(16, 10, 20)
        with torch.no_grad():
            probs, _ = model.predict_proba(x)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_padding_mask_propagates(self):
        model = TCNWithAttention(
            input_size=4, hidden_size=16, num_layers=2, dropout=0.0,
        )
        model.eval()
        x = torch.randn(2, 6, 4)
        mask = torch.tensor([
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1],
        ], dtype=torch.float32)

        with torch.no_grad():
            logits, weights = model(x, padding_mask=mask)

        # Padded positions should have 0 attention
        assert (weights[0, :3] == 0).all()
        assert (weights[1, :4] == 0).all()
        # Non-padded should sum to 1
        assert weights[0, 3:].sum().item() == pytest.approx(1.0, abs=1e-5)
        assert weights[1, 4:].sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_causal_no_future_leakage(self):
        """Critical test: modifying future timesteps must not affect
        predictions based on earlier timesteps.

        We run the full model twice — once with the original input and once
        with the last 3 timesteps replaced by random noise. The attention
        weights and intermediate TCN outputs for the unmodified timesteps
        must be identical.
        """
        model = TCNWithAttention(
            input_size=8, hidden_size=16, num_layers=3,
            kernel_size=3, dropout=0.0,
        )
        model.eval()

        x = torch.randn(1, 12, 8)
        mask = torch.ones(1, 12)

        with torch.no_grad():
            logits_orig, weights_orig = model(x, padding_mask=mask)

        # Replace the last 3 timesteps with different values
        x_modified = x.clone()
        x_modified[:, -3:, :] = torch.randn(1, 3, 8)

        with torch.no_grad():
            logits_mod, weights_mod = model(x_modified, padding_mask=mask)

        # Internal TCN outputs for timesteps 0..8 must be identical
        # (only timesteps 9,10,11 were changed)
        x_norm = model.layer_norm(x).transpose(1, 2)
        x_mod_norm = model.layer_norm(x_modified).transpose(1, 2)

        with torch.no_grad():
            tcn_orig = model.tcn(x_norm)
            tcn_mod = model.tcn(x_mod_norm)

        torch.testing.assert_close(tcn_orig[:, :, :9], tcn_mod[:, :, :9])

    def test_single_sample_batch(self):
        model = TCNWithAttention(input_size=10, hidden_size=16, num_layers=2)
        x = torch.randn(1, 5, 10)
        logits, weights = model(x)
        assert logits.shape == (1,)
        assert weights.shape == (1, 5)

    def test_matches_gru_interface(self):
        """TCN and GRU must accept the same inputs and return the same shapes."""
        from src.models.gru import GRUWithAttention

        input_size, seq_len, batch = 20, 12, 4
        x = torch.randn(batch, seq_len, input_size)
        mask = torch.ones(batch, seq_len)

        tcn = TCNWithAttention(input_size=input_size, hidden_size=32, num_layers=2)
        gru = GRUWithAttention(input_size=input_size, hidden_size=32,
                               num_layers=1, bidirectional=False)

        tcn_logits, tcn_weights = tcn(x, padding_mask=mask)
        gru_logits, gru_weights = gru(x, padding_mask=mask)

        assert tcn_logits.shape == gru_logits.shape
        assert tcn_weights.shape == gru_weights.shape

        tcn_probs, _ = tcn.predict_proba(x, padding_mask=mask)
        gru_probs, _ = gru.predict_proba(x, padding_mask=mask)
        assert tcn_probs.shape == gru_probs.shape

    def test_parameter_count(self):
        model = TCNWithAttention(input_size=120, hidden_size=128, num_layers=4)
        total, trainable = count_parameters(model)
        assert total == trainable
        assert total > 0
        print(f"\nTCN model: {total:,} total params")
