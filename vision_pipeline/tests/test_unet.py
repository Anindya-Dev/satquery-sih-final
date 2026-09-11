import torch

from vision_pipeline.models.unet_segmentor import UNetSegmentor


def test_unet_output_shape():
    torch.manual_seed(0)
    model = UNetSegmentor(in_channels=1, out_channels=1, base_channels=8)
    model.eval()

    x = torch.randn(1, 1, 120, 120)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 1, 120, 120)
