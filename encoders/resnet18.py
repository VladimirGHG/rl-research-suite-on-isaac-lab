import torch
from torch import nn
from torchvision import models

class FrozenResNet18(nn.Module):
    """
    A ResNet18 neural network wrapper, used for converting multi-hundred-thousand-element matrices to 512d vectors.
    It is a Frozen network since its weights are locked and cannot be changed to preserve the pre-trained network's stability.
    """
    def __init__(self):
        """
        FrozenResNet18 initialization with default ImageNet weights, 
        and only 9 children layers, excluding last classification layer.
        The network is in the Evaluation mode so that no weight changes happen.
        """
        super().__init__()

        # Creating the base resnet18 agent using the default ImageNet weights.
        base_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove the last child layer, destined to classify the object, since we do not need it.
        self.feature_extractor = nn.Sequential(*list(base_resnet.children())[:-1])

        # Freeze the parameters of the network.
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Explicitly set the agent to evaluation mode.
        self.feature_extractor.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The encoder's forward pass, which takes in a batch of images and outputs a batch of 512d vectors.

        Args:
            x: Raw image data to be processed.
        
        Returns:
            torch.Tensor: A batch of 512d vectors representing the features of the input images.
        """
        with torch.no_grad(): # Make sure no mathematical history is saved.
            features = self.feature_extractor(x) # Feeding the raw x image to ResNet's 9 children layers.
            return torch.flatten(features ,start_dim=1) # Flatten multidimensional array to a matrix (N_Env, 512, 1, 1) -> (N_Env, 512).
        
_GLOBAL_ENCODER = None # A flag to identify if the agent has already been laoded to a GPU memory.

def get_platform_encoder(device: str) -> FrozenResNet18:
    """
    A getter function for accessing and setting the encoder.

    Args:
        device: The GPU to which the network will be linked to.

    Returns:
        FrozenResNet18: The encoder network, which is either loaded to the GPU for the first time, 
        or already exists in the GPU memory and is returned for use.
    """
    global _GLOBAL_ENCODER
    if _GLOBAL_ENCODER is None:
        _GLOBAL_ENCODER = FrozenResNet18().to(device)
    return _GLOBAL_ENCODER