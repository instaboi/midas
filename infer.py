"""Compute depth maps for images in the input folder.
"""
import os
import glob
import time
import torch
import midas.utils as utils
import cv2
from torchvision.transforms import Compose
from midas.midas.midas_net import MidasNet
from midas.midas.transforms import Resize, NormalizeImage, PrepareForNet
from PIL import Image
import numpy as np

transform = Compose(
        [
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )
device = torch.device("cuda")






def infer(img, model_path):
    model = MidasNet(model_path, non_negative=True)
    model.to(device)
    model.eval()
    #img = utils.channel_3(img)
    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    return prediction
