import io
import logging
import os

import requests
from PIL import Image

# Define the URL of the FastAPI endpoint
url = "http://127.0.0.1:8000/generate_image/"

# Path to the image file
path = os.path.dirname(__file__)
path = os.path.dirname(path)
image_path = os.path.join(path, "inputs", "images_stormtrooper.png")

logging.debug(f"Image Path: {image_path}")

try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    image_response = requests.get(
        "https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png"
    )
    image = Image.open(io.BytesIO(image_response.content))

image_byte_arr = io.BytesIO()
image.save(image_byte_arr, format="PNG")
image = image_byte_arr.getvalue()

# Define the data to be sent in the POST request
data = {
    "text": "Spiderman, presentation, desk",  # Replace 'your_prompt_text' with the actual prompt you want to use
    "seed": 42,
}

file = {"image_file": ("images_stormtrooper.png", image, "image/png")}

# Make the POST request
logging.debug("Sending the request")
response = requests.post(url, data=data, files=file)

# Check if the request was successful
if response.status_code == 200:
    # Save the image to a file
    save_path = os.path.join(path, "outputs", "poo_generated_image.png")
    with open(save_path, "wb") as f:
        f.write(response.content)
    print("Image saved successfully.")
else:
    print(f"Failed to generate image. Status code: {response.status_code}")
    print(f"Response: {response.text}")
