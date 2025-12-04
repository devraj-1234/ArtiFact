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
    
    This loss module includes the necessary ImageNet normalization.
    """
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
        
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.feature_layers = [2, 7, 12, 21, 30]
        
        self.vgg_features = nn.ModuleList([
            vgg[:(i + 1)] for i in self.feature_layers
        ])
        
        self.criterion = nn.L1Loss()

        # --- THIS IS THE FIX ---
        # VGG was trained on ImageNet, so we must normalize the
        # [0, 1] input images to the ImageNet mean and std.
        # We register them as buffers to ensure they are moved to the
        # correct device (e.g., 'cuda') along with the model.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        # --- END OF FIX ---

    def forward(self, generated, target):
        """
        Calculates the perceptual loss.

        Args:
            generated (torch.Tensor): The output from the generator (U-Net).
                                      Expected to be in the [0, 1] range.
            target (torch.Tensor): The ground truth image.
                                   Expected to be in the [0, 1] range.

        Returns:
            torch.Tensor: The calculated perceptual loss.
        """
        
        # --- THIS IS THE FIX ---
        # Normalize both images before feeding them to VGG
        generated_norm = (generated - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        # --- END OF FIX ---

        perceptual_loss = 0.0
        for feature_extractor in self.vgg_features:
            # Feed the NORMALIZED images to the VGG
            gen_features = feature_extractor(generated_norm)
            target_features = feature_extractor(target_norm)
            perceptual_loss += self.criterion(gen_features, target_features)
            
        return perceptual_loss