from typing import Literal, Optional, Union

import torch
from diffusers import (AutoencoderKL, ControlNetModel,
                       StableDiffusionXLControlNetPipeline,
                       StableDiffusionXLPipeline)
from PIL.Image import Image


class SDXLModel:
    """A Wrapper around Diffusers pipelines for SDXL for readability and OOP"""

    def __init__(
        self,
        model_name: Literal["sdxl", "sdxl_controlnet"] = "sdxl",
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        """Create a SDXL model for generating images"""
        self.model_name = model_name
        self.pipe: Union[StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline]
        self._load_model()

        self.pipe.enable_model_cpu_offload()

        self.device = device

    def _load_model(self):
        """Private method to load the model weights from huggigngface hub"""

        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )

        if self.model_name == "sdxl":

            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                vae=self.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )

        elif self.model_name == "sdxl_controlnet":

            self.controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0",
                variant="fp16",
                use_safetensors=True,
                torch_dtype=torch.float16,
            )

            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=self.controlnet,
                vae=self.vae,
                variant="fp16",
                use_safetensors=True,
                torch_dtype=torch.float16,
            )

    def __call__(
        self,
        prompt,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        image: Optional[Image] = None,
        latent: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[torch.Tensor] = None,
        controlnet_conditioning_scale: Optional[float] = None,
    ) -> Image:
        """Generate an Image given the parameters

        Inputs:
            - prompt: The prompt to generate the image
            - image: The image to use as a controller in case of a ControlNet model
            - latent: The latent vector to generate the image from
            - ip_adapter_image: The image to generate the image from

        """

        if image is not None and self.model_name != "sdxl_controlnet":
            raise ValueError(
                "Model should be a ControlNet Model to take an image as input"
            )

        if latent is None:
            latent = torch.randn((1, 4, 128, 128), dtype=torch.float16, device="cuda")

        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image=image,  # pyright: ignore
            latent=latent,
            ip_adapter_image=ip_adapter_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            output_type="pil",
        ).images[0]

        return image
