import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from networks.paperedge import GlobalWarper, LocalWarper, WarperUtil

# Load an image, normalize its values, and convert it to the expected format for the model
def load_img(img):
    """
    Load an image, normalize its values, and convert it to the expected format for the model.

    Parameters:
        img: Input image as a NumPy array.

    Returns:
        Normalized and formatted image as a PyTorch tensor.
    """
    # Normalize pixel values to the range [0, 1]
    im = img.astype(np.float32) / 255.0
    # Reorder color channels to match the model's expected format (RGB)
    im = im[:, :, (2, 1, 0)]
    # Resize the image to the model's input size
    im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
    # Transpose the image to match the PyTorch tensor format (channels first)
    im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
    return im

# Extract paper edge from an input image using a pre-trained model
def extractPaperEdge(image, model):
    """
    Extract the paper edge from an input image using a pre-trained model.

    Parameters:
        image: Input image as a NumPy array.
        model: Path to the pre-trained model checkpoint.

    Returns:
        Extracted paper edge as a NumPy array.
    """
    # Initialize the Global Warper network
    netG = GlobalWarper().to('cuda')
    # Load the pre-trained model weights
    netG.load_state_dict(torch.load(model)['G'])
    # Set the model to evaluation mode
    netG.eval()

    # Initialize the Warper Utility and move it to the GPU
    warpUtil = WarperUtil(64).to('cuda')

    gs_d =None, # Initialize variables for edged-based deformation fields

    # Perform inference
    with torch.no_grad():
        # Load and preprocess the input image
        x = load_img(image)
        x = x.unsqueeze(0)
        x = x.to('cuda')

        # Generate the edged-based deformation field
        d = netG(x)
        d = warpUtil.global_post_warp(d, 64)
        gs_d = copy.deepcopy(d)  # Save a copy for later use

        # Resize the deformation field to match the input image size
        d = F.interpolate(d, size=256, mode='bilinear', align_corners=True)

    # Preprocess the input image for further operations
    im = image.astype(np.float32) / 255.0
    im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
    image = im.to('cuda').unsqueeze(0)

    # Resize the deformation field to match the input image size
    gs_d = F.interpolate(gs_d, (image.size(2), image.size(3)), mode='bilinear', align_corners=True)

    # Use grid sampling to apply the deformation field to the input image
    gs_y = F.grid_sample(image, gs_d.permute(0, 2, 3, 1), align_corners=True).detach()
    # Convert the result back to a NumPy array and scale to the range [0, 255]
    tmp_y = gs_y.squeeze().permute(1, 2, 0).cpu().numpy()
    return tmp_y * 255
