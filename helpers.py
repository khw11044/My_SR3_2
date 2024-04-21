from torchvision.transforms import InterpolationMode
from torchvision.transforms import transforms
import os, cv2
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchshow
import random
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SRDataset(Dataset):
    def __init__(self, dataset_path, limit = -1, _transforms = None, hr_sz = 128, lr_sz = 32) -> None:
        super().__init__()
        
        if not self.transforms:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.ColorJitter([0.5, 1]),
                transforms.RandomAdjustSharpness(1.1, p = 0.4),
                transforms.Normalize((0.5, ), (0.5,)) # normalizing image with mean, std = 0.5, 0.5
            ])

        self.hr_sz, self.lr_sz = transforms.Resize((hr_sz, hr_sz), interpolation=InterpolationMode.BICUBIC), transforms.Resize((lr_sz, lr_sz), interpolation=InterpolationMode.BICUBIC)
        
        self.dataset_path, self.limit = dataset_path, limit
        self.valid_extensions = ["jpg", "jpeg", "png", "JPEG", "JPG"]
        
        self.images_path = dataset_path
        self.images = os.listdir(self.images_path)[:self.limit]

        self.images = [os.path.join(self.images_path, image) for image in self.images if image.split(".")[-1] in self.valid_extensions]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image)
        hr_image, lr_image = self.hr_sz(image), self.lr_sz(image)

        # the core idea here is resizing the (128, 128) down to a lower resolution and then back up to (128, 128)
        return self.hr_sz(lr_image), hr_image # the hr_image is 'y' and low res image scaled to (128, 128) is our 'x' 

class MySRDataset(Dataset):
    def __init__(self, dataset_path, limit = -1, _transforms = None, hr_sz = 128, lr_sz = 32) -> None:
        super().__init__()
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.ColorJitter([0.5, 1]),
            transforms.RandomAdjustSharpness(1.1, p = 0.4),
            transforms.Normalize((0.5, ), (0.5,)) # normalizing image with mean, std = 0.5, 0.5
        ])

        self.hr_sz = transforms.Resize((hr_sz, hr_sz), interpolation=InterpolationMode.BICUBIC)
        self.lr_sz =transforms.Resize((lr_sz, lr_sz), interpolation=InterpolationMode.BICUBIC)
        
        self.dataset_path, self.limit = dataset_path, limit
        self.valid_extensions = ["jpg", "jpeg", "png", "JPEG", "JPG"]
        
        self.HR_images_path = dataset_path + '/HR'
        self.LR_images_path = dataset_path + '/LR'
        self.HR_images = os.listdir(self.HR_images_path)[:self.limit]
        self.LR_images = os.listdir(self.LR_images_path)[:self.limit]

        self.HR_images = [os.path.join(self.HR_images_path, image) for image in self.HR_images if image.split(".")[-1] in self.valid_extensions]
        self.LR_images = [os.path.join(self.LR_images_path, image) for image in self.LR_images if image.split(".")[-1] in self.valid_extensions]
        
    def __len__(self):
        return len(self.HR_images)

    def __getitem__(self, index):
        
        hr_image = cv2.imread(self.HR_images[index])
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        random.seed(index)
        torch.manual_seed(index)
        hr_image = self.transforms(hr_image)
        
        lr_image = cv2.imread(self.LR_images[index])
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        random.seed(index)
        torch.manual_seed(index)
        lr_image = self.transforms(lr_image)

        # the core idea here is resizing the (128, 128) down to a lower resolution and then back up to (128, 128)
        return self.hr_sz(lr_image), hr_image # the hr_image is 'y' and low res image scaled to (128, 128) is our 'x' 


if __name__=="__main__":
    root = './dataset'
    ds = MySRDataset(root, hr_sz = 128, lr_sz = 32)
    loader = DataLoader(ds, batch_size = 4, shuffle = True, drop_last = True, num_workers = 0)
    
    # 학습용 이미지 뽑기
    dataiter = iter(loader)
    
    for i in range(5):
        lr_image, hr_image = next(dataiter)
        torchshow.save(lr_image, f"./lr_image_{i}.jpeg")
        torchshow.save(hr_image, f"./sr_sample_{i}.jpeg")
        # torchshow.show(lr_image)
        # torchshow.show(lr_image)