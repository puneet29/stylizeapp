import torch
from PIL import Image
import io
from torchvision import transforms


def gram_matrix(tensor):
    (b, ch, h, w) = tensor.size()
    features = tensor.view(b, ch, h*w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch*h*w)
    return(gram)


def load_image(filename):
    img = Image.open(filename)
    max_size = img.size[0] if img.size[0] >= img.size[1] else img.size[1]
    scale = 1.0
    if(max_size >= 1500):
        scale = max_size/1500
    img = img.resize(
        (int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return(img)


def match_size(image, imageToMatch):
    img = image.clone().clamp(0, 255).numpy()
    img = img.squeeze(0)
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img = img.resize(
        (imageToMatch.shape[3], imageToMatch.shape[2]), Image.ANTIALIAS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    img = transform(img)
    return(img)


def normalize_batch(batch):
    # Imagenet Normalization
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


def convert_image(path, image):
    img = image.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    imgBytes = io.BytesIO()
    img.save(imgBytes, format='JPEG')
    return(imgBytes.getvalue())
