from torchvision import models 

def get_mobilenet():
    return models.mobilenet_v3_small(pretrained=True)
