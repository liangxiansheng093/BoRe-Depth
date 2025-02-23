import torch
import torch.nn as nn
import torch.nn.functional as F
from .MPViT_encoder import mpvit_tiny


class ConvBlock(nn.Module):
    """
    Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """
    Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class Conv1x1Block(nn.Module):
    """
    Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(Conv1x1Block, self).__init__()

        self.conv = self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class DeconvNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvNet, self).__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.deconv_block(x)


class LocalAttention(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super(LocalAttention, self).__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        output = self.conv_0(x)
        output = self.act(output)
        output = self.conv_1(output)
        return self.skip_add.add(output, x)


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.alpha = 10
        self.beta = 0.01

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [16, 32, 64, 128, 256]

        # decoder
        self.globalattentions = []
        self.deconvs = []
        self.localattentions = []
        self.upconvs3 = []
        self.dispconvs = []

        for i in range(4, -1, -1):
            # deconvs
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.deconvs.append(Conv1x1Block(num_ch_in, num_ch_out))

            # localattention
            num_ch_in = self.num_ch_dec[i]
            self.localattentions.append(LocalAttention(num_ch_in))

            # globalattention
            num_ch_in = self.num_ch_enc[i - 1]
            self.globalattentions.append(LocalAttention(num_ch_in))

            # upconv_3
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.upconvs3.append(ConvBlock(num_ch_in, num_ch_out))

        self.dispconvs.append(
            Conv3x3(self.num_ch_dec[0], self.num_output_channels))

        self.globalattentions = nn.ModuleList(self.globalattentions)
        self.deconvs = nn.ModuleList(self.deconvs)
        self.localattentions = nn.ModuleList(self.localattentions)
        self.upconvs3 = nn.ModuleList(self.upconvs3)
        self.dispconvs = nn.ModuleList(self.dispconvs)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        return

    def forward(self, input_features):

        self.outputs = []

        # decoder
        x = input_features[-1]

        for cnt, i in enumerate(range(4, -1, -1)):
            x = self.deconvs[cnt](x)
            x = upsample(x)
            x = [self.localattentions[cnt](x)]
            if self.use_skips and i > 0:
                y = self.globalattentions[cnt](input_features[i - 1])
                x += [y]
            x = torch.cat(x, 1)
            x = self.upconvs3[cnt](x)

            if i == 0:
                disp = self.alpha * self.sigmoid(self.dispconvs[0](x)) + self.beta
                depth = 1.0 / disp
                self.outputs.append(depth)

        self.outputs = self.outputs[::-1]
        return self.outputs


class DepthNet(nn.Module):

    def __init__(self):
        super(DepthNet, self).__init__()
        self.encoder = mpvit_tiny()
        self.num_ch_enc = [64, 96, 176, 216, 216]  # mpvit_tiny
        self.decoder = DepthDecoder(self.num_ch_enc)

    def init_weights(self):
        pass

    def forward(self, x):

        features = self.encoder(x)
        outputs = self.decoder(features)

        return outputs[0]
