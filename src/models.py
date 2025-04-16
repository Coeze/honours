# Define Capsule Network Components
import numpy as np
from .coordinates import logpolar_grid, polar_grid
from .networks import BasicCNN, EquivariantPosePredictor, TransformerCNN
from .transformers import Rotation, TransformerSequence
import torch
from torchvision.models import resnet18
import torch.nn as nn

class SelfRouting2d(nn.Module):
    def __init__(self, in_caps, out_caps, in_size, out_size, kernel_size, stride=1, padding=1, pose_out=True):
        super(SelfRouting2d, self).__init__()
        self.conv_a = nn.Conv2d(in_caps, out_caps, kernel_size, stride, padding, bias=False)
        self.conv_pose = nn.Conv2d(in_caps * in_size, out_caps * out_size, kernel_size, stride, padding, bias=False)
        self.bn_a = nn.BatchNorm2d(out_caps)
        self.bn_pose = nn.BatchNorm2d(out_caps * out_size)
        self.pose_out = pose_out

    def forward(self, a, pose, temperature=1.0):
        a_out = torch.sigmoid(self.bn_a(self.conv_a(a)))
        pose_out = self.bn_pose(self.conv_pose(pose))
        return a_out, pose_out

# Define CapsIE Model
class ETCaps(nn.Module):
    def __init__(self, in_channels, num_caps=32, caps_size=4, depth=1, final_shape=7):
        super(ETCaps, self).__init__()
        # ResNet-18 Encoder (feature extractor)
        network = BasicCNN(input_channels=in_channels, output_size=10, nf=32) 
        equivariant_transforms = TransformerSequence(
            Rotation(predictor_cls=EquivariantPosePredictor, in_channels=in_channels, nf=32, coords=polar_grid, ulim=(-np.pi / 2, np.pi / 2), vlim=(-np.pi / 2, np.pi / 2)),
            Rotation(predictor_cls=EquivariantPosePredictor, in_channels=in_channels, nf=32, coords=polar_grid, ulim=(-np.pi / 2, np.pi / 2), vlim=(-np.pi / 2, np.pi / 2)),
            Rotation(predictor_cls=EquivariantPosePredictor, in_channels=in_channels, nf=32, coords=polar_grid, ulim=(-np.pi / 2, np.pi / 2), vlim=(-np.pi / 2, np.pi / 2))
        ) 

        self.et_layer = TransformerCNN(
            net=network,
            transformer=equivariant_transforms,
            coords=logpolar_grid, 
        )

        
        self.encoder = resnet18(pretrained=False)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=1, bias=False)
        self.encoder.fc = nn.Identity()

        self.conv_a = nn.Conv2d(512, num_caps, kernel_size=1, stride=1, padding=1, bias=False)
        self.conv_pose = nn.Conv2d(512, num_caps * caps_size, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(num_caps)
        self.bn_pose = nn.BatchNorm2d(num_caps * caps_size)

        self.caps_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for _ in range(1, depth):
            self.caps_layers.append(SelfRouting2d(num_caps, num_caps, caps_size, caps_size, kernel_size=1))
            self.norm_layers.append(nn.BatchNorm2d(caps_size * num_caps))

        self.fc = SelfRouting2d(num_caps, num_caps, caps_size, caps_size, kernel_size=1, padding=0)


    def forward(self, x, temperature=0.1):
        x = self.et_layer(x)
        x = self.encoder(x)
        x = x.view(x.size(0), 512, 1, 1)
        a, pose = torch.sigmoid(self.bn_a(self.conv_a(x))), self.bn_pose(self.conv_pose(x))

        for caps_layer, bn in zip(self.caps_layers, self.norm_layers):
            a, pose = caps_layer(a, pose, 1.0)
            pose = bn(pose)

        a, pose = self.fc(a, pose, temperature)
        out = a.reshape(a.size(0), -1)
        out = out.log()
        return out
    

# Define CapsIE Model
class ResNetCaps(nn.Module):
    def __init__(self, in_channels, num_caps=16, caps_size=4, depth=3, final_shape=7):
        super(ResNetCaps, self).__init__()
        
        self.encoder = resnet18(pretrained=False)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=1, bias=False)
        self.encoder.fc = nn.Identity()

        self.conv_a = nn.Conv2d(512, num_caps, kernel_size=1, stride=1, padding=1, bias=False)
        self.conv_pose = nn.Conv2d(512, num_caps * caps_size, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(num_caps)
        self.bn_pose = nn.BatchNorm2d(num_caps * caps_size)

        self.caps_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for _ in range(1, depth):
            self.caps_layers.append(SelfRouting2d(num_caps, num_caps, caps_size, caps_size, kernel_size=1))
            self.norm_layers.append(nn.BatchNorm2d(caps_size * num_caps))

        self.fc = SelfRouting2d(num_caps, num_caps, caps_size, caps_size, kernel_size=1, padding=0)


    def forward(self, x, temperature=0.1):
        x = self.encoder(x)
        x = x.view(x.size(0), 512, 1, 1)
        a, pose = torch.sigmoid(self.bn_a(self.conv_a(x))), self.bn_pose(self.conv_pose(x))

        for caps_layer, bn in zip(self.caps_layers, self.norm_layers):
            a, pose = caps_layer(a, pose, 1.0)
            pose = bn(pose)

        a, pose = self.fc(a, pose, temperature)
        out = a.reshape(a.size(0), -1)
        return out, pose