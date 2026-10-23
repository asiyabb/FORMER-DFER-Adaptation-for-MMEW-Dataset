import torch
from torch import nn
from models.S_Former import spatial_transformer
from models.T_Former import temporal_transformer


class GenerateModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.s_former = spatial_transformer()
        self.t_former = temporal_transformer()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.s_former(x)
        x = self.t_former(x)
        return self.fc(x)