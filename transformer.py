import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm='instance'):
        """
        Params:
        - in_channels: (int) Number of channels in the input image
        - out_channels: (int) Number of channels produced by the convolution
        - kernel_size: (int or tuple) Size of the convolving kernel
        - stride: (int or tuple) Stride of the convolution
        - norm: (string, optional) Applies normalization. Accepted values: ['instance'(default), 'batch', 'None']
        """
        super(ConvLayer, self).__init__()
        # Add padding
        padding_size = kernel_size//2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride)

        # Normalization Layer
        self.norm_type = norm
        if(norm == 'instance'):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif(norm == 'batch'):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)
        assert(norm in ['instance', 'batch', 'None']
               ), 'Accepted values must belong to: "instance", "batch", "None"'

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if(self.norm_type == 'None'):
            out = x
        else:
            out = self.norm_layer(x)
        return(out)


class ResidualLayer(nn.Module):
    """
    Residual block that hops one layer.
    """

    def __init__(self, channels=128, kernel_size=3):
        """
        Params:
        - channels: (int, optional) Number of channels. Default: 128
        - kernel_size: (int or tuple, optional) Size of the convolving kernel. Default: 3
        """
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, 1)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size, 1)

    def forward(self, x):
        # preserve the residue
        residue = x
        # layer1 output + activation
        out = self.relu(self.conv1(x))
        # layer2 output
        out = self.conv2(x)
        # add residue to this output
        out = out + residue
        return(out)


class DeConvLayer(nn.Module):
    """
    Fractionally strided convolution layer or Deconvolution layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance"):
        """
        Params:
        - in_channels: (int) Number of channels in the input image
        - out_channels: (int) Number of channels produced by the convolution
        - kernel_size: (int or tuple) Size of the convolving kernel
        - stride: (int or tuple) Stride of the convolution
        - output_padding: (int or tuple) Additional size added to one side of the output shape
        - norm: (string, optional) Applies normalization. Accepted values: ['instance'(default), 'batch', 'None']
        """
        super(DeConvLayer, self).__init__()

        # Transposed Convolution or Fractional Convolution
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding_size, output_padding)

        # Normalization Layer
        self.norm_type = norm
        if (norm == "instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm == "batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)
        assert(norm in ['instance', 'batch', 'None']
               ), 'Accepted values must belong to: "instance", "batch", "None"'

    def forward(self, x):
        x = self.conv_transpose(x)
        if(self.norm_type == 'None'):
            out = x
        else:
            out = self.norm_layer(x)
        return(out)


class TransformNet(nn.Module):
    """
    Image Transform Net as described in Johnson et al
    paper: https://arxiv.org/abs/1603.08155
    """

    def __init__(self):
        """
        Conv Block -> Residual Block -> DeConv Block
        """
        super(TransformNet, self).__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3)
        )
        self.DeConvBlock = nn.Sequential(
            DeConvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeConvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm='None')
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeConvBlock(x)
        return(out)
