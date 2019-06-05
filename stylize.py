# Importing the resources
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from transformer import TransformNet
from utils import load_image, save_image


def stylize(content_image, content_scale, output_image, model, cuda=0):

    # Select GPU if available
    device = torch.device('cuda' if cuda else 'cpu')

    # Load content image
    content_image = load_image(content_image, scale=content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    # Set requires_grad to False
    with torch.no_grad():
        style_model = TransformNet()
        state_dict = torch.load(model)

        # Load the model's learnt params
        style_model.load_state_dict(state_dict)
        style_model.to(device)

        # Output image
        output = style_model(content_image).cpu()
    save_image(output_image, output[0])
