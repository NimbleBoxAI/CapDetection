import torch, time, copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms 
import os 
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes,  num_epochs=50):
    val = {"loss":[],"acc":[]}
    train = {"loss":[],"acc":[]}
    start_time=time.time()
    best_acc= 0.0
    for epoch in range(num_epochs):
        print("epoch{}/{}".format(epoch,num_epochs - 1))
        print("-" * 10) 
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            for inputs,labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "train":
              train["loss"].append(epoch_loss)
              train["acc"].append(epoch_acc.item())
            else:
              val["loss"].append(epoch_loss)
              val["acc"].append(epoch_acc.item())

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss,epoch_acc))
            if phase == "val" and epoch_acc > best_acc:
                torch.save(model,"./models/mobilenet-v3-small-best.pth")
                best_acc = epoch_acc

    time_elapsed = time.time() - start_time
    print("training completed in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed%60))
    print("best val accuracy: {:4f}".format(best_acc))
    return model


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
    model = model.load("models/mobilenet-v3-small-best.pth")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model_ft = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=5)
    torch.save(model_ft,"./models/mobilenet-v3-small-last.pth")

if __name__ == '__main__':
	main()