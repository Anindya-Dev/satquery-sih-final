from vision_pipeline.models.train_unet import train_unet


def test_train_unet_reduces_loss():
    model, losses = train_unet(epochs=10, batch_size=2, seed=0)
    assert losses[-1] < losses[0]
