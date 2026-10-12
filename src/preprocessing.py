from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


card_transform = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.ToTensor()
])


def preprocess_image(image: Image.Image):
    """
    Converts an uploaded image into a tensor
    suitable for the CNN.
    """

    image = image.convert("RGB")

    tensor = card_transform(image).unsqueeze(0)

    return tensor