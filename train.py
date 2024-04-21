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

'''
https://github.com/xwying/torchshow

코드 출처
https://medium.com/@adityanutakki6250/sr3-explained-and-implemented-in-pytorch-from-scratch-b43b9742c232
https://github.com/aditya-nutakki/SR3/tree/main


'''


test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5,)) # normalizing image with mean, std = 0.5, 0.5
            ])

def sample(model, lr_imgset_path, device = "cuda"):
    # lr_img is expected to be batched
    # set to eval mode
    model.to(device)
    model.eval()
    
    for path in tqdm(lr_imgset_path, total=len(lr_imgset_path)):
        file_name = path.split('\\')[-1]
        lr_img = cv2.imread(path)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        lr_img = test_transform(lr_img)
    
        stime = time()
        with torch.no_grad():
        
            y = torch.randn_like(lr_img, device = device)
            lr_img = lr_img.to(device)
            for i, t in enumerate(range(model.time_steps - 1, 0 , -1)):
                alpha_t, alpha_t_hat, beta_t = model.alphas[t], model.alpha_hats[t], model.betas[t]
        
                t = torch.tensor(t, device = device).long()
               
                pred_noise = model(torch.cat([lr_img, y], dim = 1), alpha_t_hat.view(-1).to(device))
                y = (torch.sqrt(1/alpha_t))*(y - (1-alpha_t)/torch.sqrt(1 - alpha_t_hat) * pred_noise)
                if t > 1:
                    noise = torch.randn_like(y)
                    y = y + torch.sqrt(beta_t) * noise
                
        ftime = time()
        torchshow.save(y, f"./save_testset/{file_name}")
        print(f"Done denoising in {ftime - stime}s ")

def train_ddpm(time_steps = 2000, epochs = 20, batch_size = 16, device = "cuda", image_dims = (3, 128, 128), low_res_dims = (3, 32, 32)):
    ddpm = DiffusionModel(time_steps = time_steps)
    c, hr_sz, _ = image_dims
    _, lr_sz, _ = low_res_dims
    
    ds = SRDataset("/mnt/d/work/datasets/nature/x128/all", hr_sz = hr_sz, lr_sz = lr_sz)
    loader = DataLoader(ds, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 2)

    opt = torch.optim.Adam(ddpm.model.parameters(), lr = 1e-3)
    criterion = nn.MSELoss(reduction="mean")

    ddpm.model.to(device)
    print()
    for ep in range(epochs):
        ddpm.model.train()
        print(f"Epoch {ep}:")
        losses = []
        stime = time()
        
        for i, (x, y) in enumerate(loader):
            
            # 'y' represents the high-resolution target image, while 'x' represents the low-resolution image to be conditioned upon.
            
            bs = y.shape[0]
            x, y = x.to(device), y.to(device)

            ts = torch.randint(low = 1, high = ddpm.time_steps, size = (bs, ))
            gamma = ddpm.alpha_hats[ts].to(device)
            ts = ts.to(device = device)

            y, target_noise = ddpm.add_noise(y, ts)
            y = torch.cat([x, y], dim = 1)
            # print(x.shape, target_noise.shape)
            # print(x.shape)
            predicted_noise = ddpm.model(y, gamma)
            loss = criterion(target_noise, predicted_noise)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            losses.append(loss.item())

            if i % 250 == 0:
                print(f"Loss: {loss.item()}; step {i}; epoch {ep}")

        ftime = time()
        print(f"Epoch trained in {ftime - stime}s; Avg loss => {sum(losses)/len(losses)}")
        
        torch.save({
                    'epoch': ep,
                    'net_dict': ddpm.state_dict()
                    }, f"./sr_ep_{ep}.pt")

        print()

        test_path = './testset'
        sample(ddpm, test_path, device = device)


if __name__ == "__main__":
    train_ddpm(time_steps=1000, epochs=20, batch_size=16)

