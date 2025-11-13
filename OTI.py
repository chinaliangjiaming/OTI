import torchvision.transforms as transforms
import torch
from PIL import Image
import torch.nn.functional as F

if __name__ == "__main__":
    img_path = "Image Path"
    object_path = "Object Mask Path"
    threshold = 0.20

    transform = transforms.Compose([transforms.ToTensor()])

    img = transform(Image.open(img_path))
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    if img.shape[0] == 4:
        img = img[:3, :, :]
    img = img.unsqueeze(0)

    object = transform(Image.open(object_path))
    object = (object > threshold) * 1.0

    kernel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.float)
    kernel_x = kernel_x.unsqueeze(0).unsqueeze(0)
    kernel_x = kernel_x.repeat(3, 1, 1, 1)

    kernel_y = torch.tensor([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]], dtype=torch.float)
    kernel_y = kernel_y.unsqueeze(0).unsqueeze(0)
    kernel_y = kernel_y.repeat(3, 1, 1, 1)

    out_x = F.conv2d(img, kernel_x, padding=1, groups=3)
    out_y = F.conv2d(img, kernel_y, padding=1, groups=3)
    out = torch.sqrt(out_x * out_x + out_y * out_y)

    _, C, H, W = out.shape
    OTI = (out * object).sum().item() / (C * H * W)
    print(f"OTI: {OTI}")
