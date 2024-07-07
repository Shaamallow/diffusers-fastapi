import torch
from diffusers import AutoencoderKL, DiffusionPipeline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    pipe.enable_model_cpu_offload()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

image = pipe(
    prompt=prompt,
    width=1024,
    height=1024,
    guidance_scale=3.5,
    num_inference_steps=30,
    output_type="pil",
).images[0]

print(image.size)
