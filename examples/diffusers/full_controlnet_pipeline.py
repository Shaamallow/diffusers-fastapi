import numpy as np
import torch
from diffusers import (AutoencoderKL, ControlNetModel,
                       StableDiffusionXLControlNetPipeline)
from diffusers.utils import load_image
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor

depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(
    "cuda"
)
feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()


def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


prompt = "spiderman doing a presentation"
image = load_image(
    "https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png"
)
controlnet_conditioning_scale = 0.5  # recommended for good generalization

depth_image = get_depth_map(image)

torch.manual_seed(42)
latent = torch.randn((1, 4, 128, 128), device="cuda")

images = pipe(
    prompt,
    latent=latent,
    image=depth_image,
    num_inference_steps=30,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
).images
images[0]

images[0].save(f"stormtrooper.png")
