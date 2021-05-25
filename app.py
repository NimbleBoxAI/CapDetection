#Imports
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


labels = ['Cap Present', 'Cap Missing']
img_mean, img_std = [0.459], [0.347]


st.title("Missing Bottlecap Detector")
st.write("""Upload pictures of Bottle for prediction, you can also upload multiple
            pictures of Bottles to predict multiple results or combine them to improve
            results.""")



st.write("Use the below checkbox for that selection before uploading images")
combine = st.checkbox("Combine images for the result")
img_list = st.file_uploader("Upload files here", accept_multiple_files=True)

models = ["EfficientNet b0", "EfficientNet b3", "MobileNet v2"]
model_choice = st.radio("Which model do you want to use ?", (models[0], models[1], models[2]))



if model_choice == models[0]:
  image_size = (224, 224)
  class EffNet(nn.Module):
    def __init__(self):
        super(EffNet, self).__init__()
        self.eff_net = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
        self.eff_net.set_swish(memory_efficient=False)
    def forward(self, x):
        x = self.eff_net(x)
        x = F.softmax(x, dim=1)
        return x
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = EffNet()
  model = torch.load("models/efnet-b0-best.pth", map_location = device)

elif model_choice == models[1]:
  image_size = (300, 300)
  class EffNet(nn.Module):
    def __init__(self):
        super(EffNet, self).__init__()
        self.eff_net = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2)
        self.eff_net.set_swish(memory_efficient=False)
    def forward(self, x):
        x = self.eff_net(x)
        x = F.softmax(x, dim=1)
        return x
        
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = EffNet()
  model = torch.load("models/efnet-b3-best-updated.pth", map_location = device)

else:
  image_size = (256, 256)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True) 
  model = torch.load("models/mobilenet-v2-best.pth", map_location = device)


model = model.to(device)
model.eval()
tfms = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        
if len(img_list) != 0:
  res = 0
  bar = st.progress(0)
  for prog, st_img in enumerate(img_list):
    img = Image.open(st_img)
    if combine:
      img = tfms(img)
      img = torch.unsqueeze(img, 0).to(device)
      res += model(img)
      bar.progress(int(prog * 100/len(img_list)) + int(100/len(img_list)))
    else:
      img = tfms(img)
      img = torch.unsqueeze(img, 0).to(device)
      res = model(img)
      bar.progress(int(prog * 100/len(img_list)) + int(100/len(img_list)))
      st.text(st_img.name + ": " + labels[torch.argmax(res)])
  res /= len(img_list)
  st.text("Predicted Class: " + labels[torch.argmax(res)])
else:
  st.text("Please Upload an image")
