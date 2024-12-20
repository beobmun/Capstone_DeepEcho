import torchvision.transforms.functional
import torch
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, video_path, video_name, transform=None):
        self.video_path = video_path
        self.video_name = video_name
        self.transform = transform
    
    def __getitem__(self, idx):
        img_path = f"{self.video_path}/{self.video_name}/{idx}.jpg"
        img = Image.open(img_path).convert('P')
        if self.transform:
            img = self.transform(img)
        img = torchvision.transforms.functional.to_tensor(img) - 0.5
        img.unsqueeze_(0)
        return img

class Resize():
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, img):
        return torchvision.transforms.functional.resize(img, self.output_size)