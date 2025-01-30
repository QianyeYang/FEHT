import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18
from torchvision.models.video import r3d_18
from icecream import ic


class R3D18_CLS(nn.Module):
    '''
    R3D18: R3D18 for video classification.
    '''
    cam_activation = None
    cam_gradients = None

    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            ):
        super(R3D18_CLS, self).__init__()
        self.model = r3d_18(weights=None)

        self.model.stem[0] = nn.Conv3d(
            in_channels,
            self.model.stem[0].out_channels,
            kernel_size=self.model.stem[0].kernel_size,
            stride=self.model.stem[0].stride,
            padding=self.model.stem[0].padding,
            bias=self.model.stem[0].bias,
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)


    def init_cam(self):
        R3D18_CLS.cam_activation = None
        R3D18_CLS.cam_gradients = None
        self.target_layer = self.model.layer4
        self.target_layer.register_forward_hook(self.save_cam_activation)
        self.target_layer.register_backward_hook(self.save_cam_gradients)


    def get_gradcam(self, x):
        '''
        Need to init_cam() before calling this function.
        '''
        self.zero_grad()
        output = self.forward(x)
        predicted_class = output.argmax(dim=1).item()
        output[0, predicted_class].backward()

        pooled_gradients = torch.mean(R3D18_CLS.cam_gradients, dim=[2, 3, 4])  # [batch_size, channel]

        R3D18_CLS.cam_activation = R3D18_CLS.cam_activation * pooled_gradients[:, :, None, None, None]
        R3D18_CLS.cam_activation = torch.sum(R3D18_CLS.cam_activation, dim=[0, 1])  # [T, H, W]
        heatmaps = F.relu(R3D18_CLS.cam_activation)  # Shape: [C, T, H, W], batch removed
        heatmaps = (heatmaps - heatmaps.min()) * 255 / (heatmaps.max() - heatmaps.min())

        # restor to the original shape
        heatmaps = F.interpolate(heatmaps[None, None, ...], size=x.shape[-3:], mode='trilinear', align_corners=False)
        heatmaps = heatmaps.detach().cpu().numpy().astype(np.uint8)[0, 0]
        
        return heatmaps


    @staticmethod
    def save_cam_gradients(module, grad_in, grad_out):
        R3D18_CLS.cam_gradients = grad_out[0]
    

    @staticmethod
    def save_cam_activation(module, input, output):
        R3D18_CLS.cam_activation = output


    def forward(self, x):
        return self.model(x)


class R3D18_WEFLP(nn.Module):
    '''
    R3D18_WEFLP: R3D18 with Each Frame Level Prediction
    '''
    def __init__(self, num_out: int):
        super(R3D18_WEFLP, self).__init__()
        self.model = r3d_18(weights=None)
        
        self.model.stem[0] = nn.Conv3d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )

        self.adjust_temporal_strides()
        self.model.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_out)


    def adjust_temporal_strides(self):
        for layer_name in ['layer2', 'layer3', 'layer4']:
            layer = getattr(self.model, layer_name)
            for block in layer:
                # Adjust conv1 stride
                if hasattr(block, 'conv1'):
                    # Access the first Conv3d layer in the Sequential
                    conv1 = block.conv1[0]
                    if isinstance(conv1, nn.Conv3d):
                        stride = conv1.stride
                        conv1.stride = (1, stride[1], stride[2])
                # Adjust downsample stride
                if block.downsample is not None:
                    downsample_conv = block.downsample[0]
                    if isinstance(downsample_conv, nn.Conv3d):
                        stride = downsample_conv.stride
                        downsample_conv.stride = (1, stride[1], stride[2])


    def forward(self, x):
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)  # Shape: (N, C, T, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # Shape: (N, C, T)
        x = x.permute(0, 2, 1)  # Shape: (N, T, C)
        x = self.model.fc(x)  # shape (N, T, num_out)

        pred_classes = F.softmax(x[..., :2], dim=-1)
        pred_center = x[..., 2:4]
        pred_radius = x[..., 4:]
        return pred_classes, pred_center, pred_radius


class R2D18_WEFLP(nn.Module):
    """
    A 2D ResNet18 variant that produces per-frame outputs:
      - pred_classes: shape (N, T, 2)  after softmax
      - pred_center:  shape (N, T, 2)
      - pred_radius:  shape (N, T, ?)
    """
    def __init__(self, channel_in: int, num_out: int):
        super(R2D18_WEFLP, self).__init__()

        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(
            in_channels=channel_in, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_out * channel_in)


    def forward(self, x):
        N, C, T, H, W = x.shape
        assert C==1, "Input tensor should have 1 channel to use this network."
        x = x[:, 0, ...]
        x = self.model(x)
        
        x = x.view(N, T, -1)
        pred_classes = F.softmax(x[..., :2], dim=-1)  # e.g. shape (N, T, 2)
        pred_center = x[..., 2:4]                    # e.g. shape (N, T, 2)
        pred_radius = x[..., 4:]                     # e.g. shape (N, T, ?)
        
        return pred_classes, pred_center, pred_radius



if __name__ == "__main__":

    data = torch.randn(1, 1, 64, 224, 224)
    # model = R3D18_WEFLP(6)
    model = R2D18_WEFLP(64, 6)
    output = model(data)
    for x in output:
        print(x.shape)

    # model = R3D18(1, 2)
    # output = model(data)
    # print(output.shape)