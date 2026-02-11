import pytest
import torch
from exp.models.components.stacked_features import (
    LerpStackedFeatures,
    ToStackedFeatures,
)
class TestLerpStackedFeatures:
    @pytest.mark.parametrize(
        "dim_in,dim_out,num_stack",
        [
            (128, 64, 8),
            (32, 16, 5),
        ],
    )
    @pytest.mark.parametrize(
        "batch_shape",
        [
            (),
            (4,),
            (3, 4),
            (2, 3, 5),
        ],
    )
    def test_forward(self, batch_shape, dim_in, dim_out, num_stack):
        mod = LerpStackedFeatures(dim_in, dim_out, num_stack)
        input_shape = batch_shape + (num_stack, dim_in)
        expected_shape = batch_shape + (dim_out,) if batch_shape else (dim_out,)
        feature = torch.randn(*input_shape)
        out = mod.forward(feature)
        assert out.shape == expected_shape
class TestToStackedFeatures:
    @pytest.mark.parametrize(
        "dim_in,dim_out,num_stack",
        [
            (64, 128, 4),
            (32, 16, 5),
        ],
    )
    @pytest.mark.parametrize(
        "batch_shape",
        [
            (),
            (8,),
            (3, 8),
            (2, 4, 6),
        ],
    )
    def test_forward(self, batch_shape, dim_in, dim_out, num_stack):
        mod = ToStackedFeatures(dim_in, dim_out, num_stack)
        input_shape = batch_shape + (dim_in,)
        expected_shape = (
            batch_shape + (num_stack, dim_out) if batch_shape else (num_stack, dim_out)
        )
        feature = torch.randn(*input_shape)
        out = mod.forward(feature)
        assert out.shape == expected_shape
