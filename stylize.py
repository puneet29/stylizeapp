# Importing the resources
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from transformer import TransformNet
from utils import convert_image, load_image, match_size


def stylize(content_image, output_image, model, style_strength, cuda=0):

    # Select GPU if available
    device = torch.device('cuda' if cuda else 'cpu')

    # Load content image
    input_image = load_image(content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    input_image = content_transform(input_image)
    input_image = input_image.unsqueeze(0).to(device)

    # Set requires_grad to False
    with torch.no_grad():
        style_model = TransformNet()
        state_dict = torch.load(model)

        # Load the model's learnt params
        style_model.load_state_dict(state_dict)
        style_model.to(device)

        # Output image
        output = style_model(input_image).cpu()

    # Make sure both images have same shapes
    content_image = match_size(input_image, output)
    weighted_output = output * style_strength + \
        (content_image * (1 - style_strength))
    return(convert_image(output_image, weighted_output[0]))
