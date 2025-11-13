import torchvision.transforms as transforms
from PIL import Image


if __name__ == "__main__":
    object_path = "Object Mask Path"
    threshold = 0.20

    transform = transforms.Compose([transforms.ToTensor()])
    object = transform(Image.open(object_path))
    C, H, W = object.shape

    object = (object > threshold) * 1.0

    OAR = object.sum().item() / (C * H * W)
    print(f"OAR: {OAR}")
