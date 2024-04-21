import cv2
import glob
from tqdm import tqdm
root = './testset/test'
new_root = './testset/new_test'
all_files = sorted(glob.glob(root + '/*'))

for file in tqdm(all_files, total=len(all_files)):
    new_name = new_root + '/' + file.split('\\')[-1].split('.')[0]
    src=cv2.imread(file, cv2.IMREAD_COLOR)
    H,W,_ = src.shape
    new_size = int(H//2)
    dst=src.copy()


    dst1=src[0:new_size, 0:new_size]
    dst2=src[0:new_size, new_size:new_size*2]  
    dst3=src[new_size:new_size*2, 0:new_size]
    dst4=src[new_size:new_size*2, new_size:new_size*2]
    
    cv2.imwrite('{}_1.png'.format(new_name),dst1) 
    cv2.imwrite('{}_2.png'.format(new_name),dst2) 
    cv2.imwrite('{}_3.png'.format(new_name),dst3) 
    cv2.imwrite('{}_4.png'.format(new_name),dst4) 