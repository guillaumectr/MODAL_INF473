import torch.nn as nn
import torch
import torchvision
from models.resnet import ResNetFinetune
from models.dinov2 import DinoV2Finetune

class SimpleDouble(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.dinov2.head = nn.Identity()
        if frozen:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.dinov2.parameters():
                param.requires_grad = False
            if unfreeze_last_layer:
                for param in self.dinov2.norm.parameters():
                    param.requires_grad = True
                for param in self.dinov2.blocks[-1].parameters():
                    param.requires_grad = True
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(self.dinov2.norm.normalized_shape[0]+2048, 2048), nn.ReLU(), nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, num_classes), nn.Softmax())

    def forward(self, x):
        y = self.resnet(x)
        x = self.dinov2(x)
        x = torch.cat((x,y), dim=-1)
        x = self.classifier(x)
        return x