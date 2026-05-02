import torch.nn as nn
from torchvision import models

def get_resnet():
    model = models.resnet18(pretrained=True)

    # 改成二分类
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model