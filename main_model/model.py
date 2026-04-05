from .unet import UNet


def init_model(model_type="Unet"):
    if model_type.lower() == "unet":
        return UNet()

    raise ValueError(
        f"Unknown model type: {model_type}. Supported model_type values: Unet"
    )
