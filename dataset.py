import json
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms

class NordsonDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.transform = transform
        self.class_to_id = json.load(open(os.path.join(main_dir,'class_map.json')))
        self.id_to_class = {self.class_to_id[c]:c for c in self.class_to_id}
        self._img_paths = []
        self._labels = []
        #load image path with its label
        #assign each 
        dirs = os.listdir(main_dir)
        for d in dirs:
            p = os.path.join(main_dir, d)
            if not os.path.isdir(p):
                continue
            assert d in self.class_to_id
            label = [self.class_to_id[d]]
            l = glob.glob(os.path.join(p,'*.png'))
            self._img_paths += l
            self._labels += label*len(l)


    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        img_loc = self._img_paths[idx]
        img_name = os.path.basename(img_loc)
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        label = self._labels[idx]
        return tensor_image, label, img_name


if __name__ == '__main__':
    path_data = './data/cropped_224x224'
    tfms = transforms.Compose([transforms.Resize((224,224)), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    dataset = NordsonDataSet(path_data, transform=tfms)
    print(len(dataset))
    dataset.__getitem__(100)
    