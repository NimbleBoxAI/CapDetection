#imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms 
import os 
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from model import train_model, get_model
from infer import transform_img, predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(torch.utils.data.Dataset):
  def __init__(self, root_path, transforms=None ):
    self.segment_net = get_model()
    self.root_path = root_path
    self.transform_img = transforms
    self.img_list = []
    for phase in ["Capped", "NoCap"]:
        img_root = os.path.join(root_path, phase)
        for img_name in os.listdir(img_root):
            img_path = os.path.join(img_root, img_name)
            self.img_list.append([img_path, phase])
        
  def __len__(self):
    return len(self.img_list)
             
  def __getitem__(self, index):
    img = Image.open(self.img_list[index][0])
    label = 0 if self.img_list[index][1] == "Capped" else 1
    img_t = transform_img(img).unsqueeze(0)
    img_seg = predict([img], img_t, self.segment_net)
    if len(img_seg[0]) == 0:
      img_seg = img
    else:
      img_seg = img_seg[0][0]
      img_seg = Image.fromarray(img_seg)

    img_transformed = self.transform_img(img_seg) 
    return img_transformed, label

def load_data(data_dir, data_transforms):
    dataloaders = {}
    train_dataset = Dataset("Dataset/train", data_transforms["train"])
    test_dataset = Dataset("Dataset/val", data_transforms["val"])
    dataloaders["train"] = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2) 
    dataloaders["val"] = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2) 
    class_names=["Capped",  "NoCap"]
    dataset_sizes = {"train": len(train_dataset), "val": len(test_dataset)}
    print(class_names)
    return dataloaders, dataset_sizes 


def get_transforms(image_size=(224,224)):
    data_transforms = {
                        "train": transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.ColorJitter(hue=.05, saturation=.05),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(20, resample=Image.BILINEAR),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                        "val": transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                        }
    return data_transforms


def main():
    transforms = get_transforms()
    dataloaders, dataset_sizes = load_data(data_dir = "Dataset/", data_transforms= transforms)
    model = get_model()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model_ft = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=5)
    torch.save(model_ft.state_dict(),"./models/mobilenet-v3-small-last.pth")

if __name__ == '__main__':
	main()