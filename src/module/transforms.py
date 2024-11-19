import torch
import torch.nn as nn
import random

class Cutout(object):
    def __init__(self, h, w):
        """
        Initializes the Cutout transform.

        Args:
            h (int): Height of the cutout region.
            w (int): Width of the cutout region.
        """
        self.h = h
        self.w = w

    def __call__(self, x):
        """
        Applies Cutout on the input image.

        Args:
            x (Tensor): Input image tensor of shape (C, H, W).

        Returns:
            Tensor: Image with a randomly cut-out region.
        """
        *_, H, W = x.size()  # Get image dimensions

        # Ensure the cutout region fits within the image dimensions
        x0 = random.randint(0, W - self.w)
        y0 = random.randint(0, H - self.h)

        # Set the cutout region to zero
        x[:, y0:y0 + self.h, x0:x0 + self.w] = 0.0

        return x
