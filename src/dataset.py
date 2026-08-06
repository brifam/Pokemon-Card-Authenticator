### Code for Transforms, ImageFolder, and Dataloader



#import libraries

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader

BATCH_SIZE = 32

g = torch.Generator()
g.manual_seed(42)

card_transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor()])


train_dataset = datasets.ImageFolder(root = "dataset/train", transform= card_transform)

train_size = int(len(train_dataset) * .8)
validation_size = len(train_dataset) - train_size


train_set, validation_set = random_split(train_dataset,[train_size,validation_size])

train_dataloader = DataLoader(dataset = train_set, batch_size= BATCH_SIZE, shuffle= True, generator= g)
validation_dataloader = DataLoader(dataset = validation_set, batch_size = BATCH_SIZE, shuffle = False)


#Check if successful
if __name__ == "__main__":
    print(f"Total amount of dataset images: {len(train_dataset)}")
    print(f"Training set Size: {train_size}")
    print(f"Validation set Size: {validation_size}")
    print(f"Training set Amount: {len(train_dataloader)} batches")
    print(f"Validation set Amount: {len(validation_dataloader)} batches")
