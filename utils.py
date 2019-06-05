import torch
from PIL import Image


def gram_matrix(tensor):
    (b, ch, h, w) = tensor.size()
    features = tensor.view(b, ch, h*w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch*h*w)
    return(gram)


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if(size is not None):
        img = img.resize((size, size), Image.ANTIALIAS)
    if(scale is not None):
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)),
                         Image.ANTIALIAS)
    return(img)


def normalize_batch(batch):
    # Imagenet Normalization
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


def save_image(path, image):
    img = image.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(path)
