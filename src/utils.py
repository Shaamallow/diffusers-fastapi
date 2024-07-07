import io
import logging

from PIL import Image
from PIL.Image import Image as ImageType


def decode_image(encoded_image: bytes) -> ImageType:

    logging.debug("Decoding Image")
    image = Image.open(io.BytesIO(encoded_image)).convert("RGB")

    return image


def encode_image(image: ImageType) -> bytes:

    logging.debug("Encoding Image")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()
