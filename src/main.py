import logging

import torch
from diffusers.utils import load_image
from fastapi import FastAPI, File, Form, Response, UploadFile
from PIL.Image import Image

from models.sdxl_model import SDXLModel
from processors.depth_processor import ControlNetDepthProcessor
from utils import decode_image, encode_image

app = FastAPI()

model = SDXLModel(model_name="sdxl_controlnet")
processor = ControlNetDepthProcessor()

logging.basicConfig(level=logging.DEBUG)


@app.get("/")
def read_root():
    return {"Hello": "World"}


# Call using Post request to generate an Image
@app.post(
    "/generate_image/",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def generate_image(
    text: str = Form(default=""),
    seed: int = Form(default=42),
    image_file: UploadFile = File(...),
):

    # Load controlNet Image for depth model

    image_data = await image_file.read()
    logging.debug(f"Recieved image type : {type(image_data)}")

    image = decode_image(image_data)
    logging.debug(f"Decoded the image with shape: {image.size}")

    control_image = processor(image)

    torch.manual_seed(seed)
    latent = torch.randn((1, 4, 128, 128), device="cuda")

    logging.debug("Generating the Image")
    image: Image = model(
        prompt=text,
        latent=latent,
        image=control_image,
        guidance_scale=3.5,
        num_inference_steps=30,
        controlnet_conditioning_scale=0.5,
    )

    # Get the bytes of the image
    img_byte_arr = encode_image(image)

    # Return a Response object, which will be an image
    return Response(content=img_byte_arr, media_type="image/png")
