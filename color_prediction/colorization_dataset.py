import os
import requests
import zipfile
import shutil
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        split_dir = os.path.join(self.root_dir, self.split)
        image_paths = []
        if self.split == 'train':
            for class_dir in os.listdir(split_dir):
                images_dir = os.path.join(split_dir, class_dir, 'images')
                for img_name in os.listdir(images_dir):
                    image_paths.append(os.path.join(images_dir, img_name))
        elif self.split == 'val':
            val_images_dir = os.path.join(split_dir, 'images')
            for img_name in os.listdir(val_images_dir):
                image_paths.append(os.path.join(val_images_dir, img_name))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        image = np.array(image)
        lab_image = rgb2lab(image).astype("float32")
        lab_image = transforms.ToTensor()(lab_image)
        
        # Normalize the L-channel to be between -1 and 1
        L_channel = lab_image[[0], ...] / 50.0 - 1.0
        # Normalize the a and b channels to be between -1 and 1
        ab_channels = lab_image[[1, 2], ...] / 128.0
        
        return {'L': L_channel, 'ab': ab_channels}

def prepare_train_and_val_data():
    if not os.path.exists("tiny-imagenet-200"):
        print("Downloading Tiny ImageNet dataset...")
        zip_file_path = "tiny-imagenet-200.zip"
        
        # Using a reliable download method with requests
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        response = requests.get(url, stream=True)
        with open(zip_file_path, "wb") as file:
            shutil.copyfileobj(response.raw, file)

        print("Unzipping the dataset...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        os.remove(zip_file_path)
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Create datasets
    train_dataset = ColorizationDataset(root_dir="tiny-imagenet-200", split='train', transform=transform)
    val_dataset = ColorizationDataset(root_dir="tiny-imagenet-200", split='val', transform=transform)
    
    return train_dataset, val_dataset