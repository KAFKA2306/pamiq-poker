import pytest
import torch
from exp.models.components.image_patchifier import ImagePatchifier
class TestImagePatchifier:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("img_size", [224, 512])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize("embed_dim", [768])
    def test_forward(self, batch_size, img_size, patch_size, embed_dim):
        layer = ImagePatchifier(patch_size, 3, embed_dim)
        image = torch.randn(batch_size, 3, img_size, img_size)
        out = layer(image)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (batch_size, (img_size // patch_size) ** 2, embed_dim)
    @pytest.mark.parametrize(
        "image_size,patch_size,embed_dim,batch_size",
        [
            (64, 8, 768, 1),
            (224, 16, 512, 2),
            (256, 32, 384, 3),
            ((128, 256), (16, 32), 768, 1),
            ((96, 96), (12, 12), 256, 4),
        ],
    )
    def test_compute_num_patches_consistency_with_forward(
        self, image_size, patch_size, embed_dim: int, batch_size: int
    ):
        """Test that compute_num_patches matches actual forward pass output."""
        expected_patches_2d = ImagePatchifier.compute_num_patches(
            image_size, patch_size
        )
        expected_total_patches = expected_patches_2d[0] * expected_patches_2d[1]
        patchifier = ImagePatchifier(patch_size=patch_size, embed_dim=embed_dim)
        if isinstance(image_size, tuple):
            height, width = image_size
        else:
            height = width = image_size
        input_image = torch.randn(batch_size, 3, height, width)
        output = patchifier(input_image)
        assert output.shape[1] == expected_total_patches
        assert output.shape == (batch_size, expected_total_patches, embed_dim)
