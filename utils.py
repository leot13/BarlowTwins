import numpy as np
import torch

#data augmentation libraries
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

#torch dataset library
from torch.utils.data import Dataset


class CIFARDataset(Dataset):
  def __init__(self, train_data, transform1= None, transform2= None):

    self.train_data = train_data
    self.transform1 = transform1
    self.transform2 = transform2

  def __len__(self):
    return len(self.train_data)

  def __getitem__(self, idx):
    label = self.train_data[idx][1]

    image = np.array(self.train_data[idx][0])
    image1 = image
    image2 = image
    
    #Transform the same image with 2 different transforms
    if self.transform1 is not None:
      image1 = self.transform1(image = image)['image']

    if self.transform2 is not None:
      image2 = self.transform2(image = image)['image']
    

    return image1, image2, label



def makeTransforms(IMG_HEIGHT, IMG_WIDTH):
    
    # Here, A.RandomBrightnessContrast and A.HueSaturationValue replace the ColorJitter from the paper 
    # because the torchvision.transform implementation of ColorJitter is different from the Albumentation one 
    transform1 = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.RandomResizedCrop (height= IMG_HEIGHT, width=IMG_WIDTH, scale=(0.08, 1.0), interpolation=cv2.INTER_CUBIC, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(hue_shift_limit=int(0.1 * 180),
                                 sat_shift_limit=int(0.2 * 255),
                                 val_shift_limit=0, p=0.8),
            A.ToGray(p=0.2),
            A.Solarize (p=0.0),
            A.GaussianBlur(sigma_limit=[0.1, 0.2], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
    )

    transform2 = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.RandomResizedCrop (height= IMG_HEIGHT, width=IMG_WIDTH, scale=(0.08, 1.0), interpolation=cv2.INTER_CUBIC, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(hue_shift_limit=int(0.1 * 180),
                                 sat_shift_limit=int(0.2 * 255),
                                 val_shift_limit=0, p=0.8),
            A.ToGray(p=0.2),
            A.Solarize (p=0.2),
            A.GaussianBlur( sigma_limit=[0.1, 0.2], p=0.1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
    )

    return transform1, transform2


def makeTransforms_Fine_Tuning(IMG_HEIGHT, IMG_WIDTH):

    transform1 = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
    )

    transform2 = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
    )

    return transform1, transform2


def compute_accuracy(out, labels):
  accuracy = 0

  for (out_vec, label) in zip(out, labels):
    pred = torch.argmax(out_vec)
    if pred == label:
      accuracy += 1

  return accuracy/out.shape[0]


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])