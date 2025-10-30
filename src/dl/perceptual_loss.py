"""
perceptual_loss.py

Defines a perceptual loss function using a pre-trained VGG19 network.
This loss is used to train the U-Net to produce more visually realistic results
by comparing high-level features instead of just pixel values.
"""

import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class PerceptualLoss(nn.Module):
    """
    Calculates the perceptual loss, which is the L1 loss between the
    feature maps of a generated image and a target image, extracted from
    a pre-trained VGG19 network.
    """
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG19 model and send it to the specified device
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
        
        # We don't want to train the VGG model, so we freeze its parameters
        for param in vgg.parameters():
            param.requires_grad = False
            
        # Define the layers to be used for feature extraction
        # These are standard layers used for perceptual loss in literature
        self.feature_layers = [2, 7, 12, 21, 30]
        
        # Create a list of sequential models up to each feature layer
        self.vgg_features = nn.ModuleList([
            vgg[:(i + 1)] for i in self.feature_layers
        ])
        
        self.criterion = nn.L1Loss()

    def forward(self, generated, target):
        """
        Calculates the perceptual loss.

        Args:
            generated (torch.Tensor): The output from the generator (U-Net).
            target (torch.Tensor): The ground truth image.

        Returns:
            torch.Tensor: The calculated perceptual loss.
        """
        perceptual_loss = 0.0
        for feature_extractor in self.vgg_features:
            gen_features = feature_extractor(generated)
            target_features = feature_extractor(target)
            perceptual_loss += self.criterion(gen_features, target_features)
            
        return perceptual_loss
