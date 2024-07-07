import torch.nn as nn
from einops.einops import rearrange
from .pos_enc import ImgPosEnc
from .mlp_mixer import Block
import copy


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, embed_dim: int, patch_size=8, in_chans=1):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 8 == 0 and W % 8 == 0, \
            f"Input image size ({H}、{W}) doesn't match model."
        x = self.proj(x)  # [4,256,h/8,w/8]
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DCTNet(nn.Module):
    def __init__(self, d_model, num_mlp_layers):
        super(DCTNet, self).__init__()
        self.patch_embed = PatchEmbed(embed_dim=d_model, patch_size=8, in_chans=1)
        self.out_channel = d_model
        # self.conv_layers = VGG1()
        self.channel_att = ChannelAtt(d_model, 16)
        self.pos_enc_2d = ImgPosEnc(256, normalize=True)
        self.norm = nn.LayerNorm(256)
        self.mlps = _get_clones(Block(d_model=d_model), num_mlp_layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x, mask):  # [4, 1, H, W]
        x = self.patch_embed(x)  # [4,256,H//8,W//8]
        x_mlp = rearrange(x, "b d h w -> b h w d")
        for i, mod in enumerate(self.mlps):
            x_mlp = mod(x_mlp)
        x = rearrange(x_mlp, "b h w d -> b d h w")
        # Apply ChannelAttention
        x = self.channel_att(x)
        x = self.pool(x)
        x = rearrange(x, "b d h w -> b h w d")

        x = self.pos_enc_2d(x, mask)
        x = self.norm(x)

        return x


class ChannelAtt(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
