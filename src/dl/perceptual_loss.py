import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()

        # Load pretrained VGG19 features to device
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device)

        # Use first 16 layers
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:16]).eval()

        # Freeze VGG parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        # Register mean/std buffers (added .to(device) FIX)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
        )

        self.l1 = nn.L1Loss()

    def preprocess_for_vgg(self, img):
        """
        Convert Tanh output [-1,1] -> [0,1], then apply ImageNet normalization
        """

        # Convert from [-1,1] to [0,1]
        img = (img + 1) / 2

        # Ensure mean/std are on same device as img (safety for DDP/multi-gpu)
        mean = self.mean.to(img.device)
        std = self.std.to(img.device)

        # Normalize
        img = (img - mean) / std
        return img

    def forward(self, generated, target):
        # Preprocess
        gen_vgg = self.preprocess_for_vgg(generated)
        tgt_vgg = self.preprocess_for_vgg(target)

        # Extract features
        gen_features = self.vgg_layers(gen_vgg)

        # DO NOT track gradients for target
        with torch.no_grad():
            tgt_features = self.vgg_layers(tgt_vgg)

        # L1 loss between feature maps
        return self.l1(gen_features, tgt_features)
