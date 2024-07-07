import numpy as np
import torch
from diffusers.utils import load_image

from src.models.sdxl_model import SDXLModel
from src.processors.depth_processor import ControlNetDepthProcessor

sd_model = SDXLModel(model_name="sdxl_controlnet")
processor = ControlNetDepthProcessor()


prompt = "stormtrooper lecture, photorealistic"
image = load_image(
    "https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png"
)
controlnet_conditioning_scale = 0.5  # recommended for good generalization

depth_image = processor(image)

# Fix the seed
# torch.manual_seed(42)
for i in range(1):

    latent = torch.randn((1, 4, 128, 128), device="cuda")

    images = sd_model(
        prompt=prompt,
        image=depth_image,
        latent=latent,
        guidance_scale=controlnet_conditioning_scale,
        num_inference_steps=30,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    )

    images.save(f"outputs/stromtrooper_grid.png")
