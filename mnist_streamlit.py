import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os

class SlimAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,kernel_size = 3, stride =1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size =3, stride = 2),
            nn.Conv2d(32,64,kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size =3, stride = 2),
            nn.Conv2d(64,128,kernel_size = 3, padding =1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128,256,kernel_size = 3, padding =1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256,128,kernel_size = 3, padding =1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size =3, stride = 2),
            )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128,1024),
            nn.ReLU(inplace= True),
            nn.Dropout(),
            nn.Linear(1024,1024),
            nn.ReLU(inplace= True),
            nn.Linear(1024,num_classes)
            )
    def forward(self,x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

def predict(im):
    inputs = transform(im)
    inputs = inputs.to(device)
    results = net(inputs.unsqueeze(0)).argmax(dim=1).to('cpu').numpy()
    return results[0]


transform = transforms.Compose([transforms.ToTensor()])
net = SlimAlexNet(num_classes=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
net.load_state_dict(torch.load('best_model.pt', map_location=lambda storage, loc: storage))



#streamlit start
st.title("Welcome to Handwritten Digits Preditor")
st.markdown("This application is a Streamlit dashboard that can be used "
            "to predict 0-9  🗽💥🚗")

input_buffer = st.file_uploader("Upload a handwritten digit", type=("png", "jpg"))

if st.button("Predict"):
    im = Image.open(input_buffer).convert('L')
    im = im.resize((28,28),Image.NEAREST)
    result = predict(im)
    rsl = 'Congratulations! Your uploaded digit predicted as: %d'%result
    st.text(rsl)
    st.balloons()
