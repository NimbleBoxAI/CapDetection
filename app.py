#Imports
from os import path
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from infer import get_model, predict, transform_img
import numpy as np



def get_mobilenet(path, device):
  image_size = (256, 256)
  model = torch.load(path, map_location = device)
  model = model.to(device)
  return model

def get_transforms(image_size):
  tfms = transforms.Compose([
                  transforms.Resize(image_size),
                  transforms.ToTensor(),
                  transforms.Normalize(
                          [0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
                          ])
  return tfms


def show_result(img_list, tfms, device):
  labels = ['Cap Present', 'Cap Missing']
  segment_net = get_model()
  model = get_mobilenet(path = "models/mobilenet-v3-small-best.pth", device = device)
  model.eval()
  if len(img_list) != 0:
    res = 0
    bar = st.progress(0)
    for prog, st_img in enumerate(img_list):
      img = Image.open(st_img).convert('RGB')
      st.image(np.array(img))
      img_t = transform_img(img).unsqueeze(0)
      predictions = predict([img], img_t, segment_net)[0]
      for j in predictions:
        img = Image.fromarray(j)
        img_t = tfms(img)
        img_t = torch.unsqueeze(img_t, 0).to(device)
        res = model(img_t)
        bar.progress(int(prog * 100/len(img_list)) + int(100/len(img_list)))
        st.image(img)
        st.text("Label: " + labels[torch.argmax(res)])
  else:
    st.text("Please Upload an image")


def main():
  st.title("Missing Bottlecap Detector")
  st.write("""Upload pictures of Bottle for prediction, you can also upload multiple
            pictures of Bottles to predict multiple results or combine them to improve
            results.""")
  st.write("Use the below checkbox for that selection before uploading images")
  combine = st.checkbox("Combine images for the result")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  img_list = st.file_uploader("Upload files here", accept_multiple_files=True)
  tfms = get_transforms(image_size=(256,256))
  show_result(img_list, tfms, device)

  
if __name__ == '__main__':
	main()