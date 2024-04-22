import torchshow
from modules import *
from helpers import *
import helpers
from time import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import glob

'''
https://github.com/xwying/torchshow

코드 출처
https://medium.com/@adityanutakki6250/sr3-explained-and-implemented-in-pytorch-from-scratch-b43b9742c232
https://github.com/aditya-nutakki/SR3/tree/main


'''

test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5,)), # normalizing image with mean, std = 0.5, 0.5
                transforms.Resize((128, 128), interpolation=InterpolationMode.BICUBIC)
            ])

def sample(model, lr_imgset_path, device = "cuda"):
    # lr_img is expected to be batched
    # set to eval mode
    model.to(device)
    model.eval()
    
    lr_imgset_path = os.listdir(lr_imgset_path)
    for fname in tqdm(lr_imgset_path, total=len(lr_imgset_path), desc='Test'):
        path = f"./testset/{fname}"
        lr_img = cv2.imread(path)
        h,w,c=lr_img.shape
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        lr_img = test_transform(lr_img).reshape(1,c,h*4,w*4)

        #stime = time()
        with torch.no_grad():
        
            y = torch.randn_like(lr_img, device = device)
            lr_img = lr_img.to(device)
            for i, t in enumerate(tqdm(range(model.time_steps - 1, 0 , -1))):         # refining
                alpha_t, alpha_t_hat, beta_t = model.alphas[t], model.alpha_hats[t], model.betas[t]
        
                t = torch.tensor(t, device = device).long()
               
                pred_noise = model(torch.cat([lr_img, y], dim = 1), alpha_t_hat.view(-1).to(device))    # ([1, 3, 32, 32])
                y = (torch.sqrt(1/alpha_t))*(y - (1-alpha_t)/torch.sqrt(1 - alpha_t_hat) * pred_noise)  # ([1, 3, 32, 32])
                if t > 1:
                    noise = torch.randn_like(y)
                    y = y + torch.sqrt(beta_t) * noise
                
        #ftime = time()

        torchshow.save(y, f"./save_testset/{fname}")
        #print(f"Done denoising in {ftime - stime}s ")
        
def train_ddpm(model_load=True, time_steps = 2000, epochs = 20, batch_size = 16, device = "cuda", image_dims = (3, 128, 128), low_res_dims = (3, 32, 32), dataset_path = './traindataset'):
    ddpm = DiffusionModel(time_steps = time_steps)
    start_epoch = 0
    if model_load:
        checkpoint_file = "./sr_ep_19.pt"
        checkpoint = torch.load(checkpoint_file, map_location=device)
        ddpm.model.load_state_dict(checkpoint["net_dict"])
        start_epoch = checkpoint["epoch"]
    
    c, hr_sz, _ = image_dims
    _, lr_sz, _ = low_res_dims
    

    ddpm.model.to(device)
    print()

        
    print('test')
    lr_imgset_path = './testset'
    sample(ddpm, lr_imgset_path, device = device)
    




if __name__ == "__main__":
    root = './dataset'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    train_ddpm(model_load=True, time_steps=1000, epochs=20, batch_size=16, device = DEVICE, image_dims = (3, 128, 128), low_res_dims = (3, 32, 32), dataset_path=root)

