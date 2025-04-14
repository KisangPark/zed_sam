import torch
from torchvision.models import alexnet

model = alexnet(pretrained=True).eval().cuda()
x = torch.ones((1, 3, 224, 224)).cuda()

try:
    model(x)
except Exception as e:
    print("error running model on gpu:", e)