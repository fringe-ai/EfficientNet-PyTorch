#%%
import glob
import os
import cv2
from image_utils.Rosenbrock import resize 
from image_utils.pad_to_size import pad_array

#%%
src_data_dir='./data/cropped'
dst_data_dir='./data/cropped_224x224'
DIM=224

files=glob.glob(os.path.join(src_data_dir,'**/*.png'),recursive=True)

#%%
for file in files:
    img=cv2.imread(file)
    img=resize(img, width=DIM)
    img=pad_array(img,DIM,DIM)
    fout=os.path.split(file)[1]
    fout=os.path.splitext(fout)[0]+f'_{DIM}x{DIM}.png'
    pout=os.path.split(file)[0]
    pout=os.path.split(pout)[1]
    pout=os.path.join(dst_data_dir,pout)
    if not os.path.isdir(pout):
        os.makedirs(pout)
    out_name = os.path.join(pout, fout)
    cv2.imwrite(out_name,img)





# %%
