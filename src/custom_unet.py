from diffusers import UNet2DModel

def get_ldm_unet():
    return UNet2DModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )