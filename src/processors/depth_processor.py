from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as IMAGE
from transformers import DPTForDepthEstimation, DPTImageProcessor


class ControlNetDepthProcessor:
    """A Wrapper around DPTFor Depth Estimation"""

    def __init__(self):
        """Create a ControlNetDepthProcessor model for generating depth images"""

        self.preprocessor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")

    def __call__(
        self,
        image: IMAGE,
        size: Union[None, Tuple[int, int]] = None,
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        """Generate a depth image from a given image"""

        if size is None:
            size = (1024, 1024)

        image = self.preprocessor(images=image, return_tensors="pt").pixel_values

        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.model(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image_tensor = torch.cat([depth_map] * 3, dim=1)

        image_tensor = image_tensor.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image_tensor * 255.0).clip(0, 255).astype(np.uint8))

        return image
