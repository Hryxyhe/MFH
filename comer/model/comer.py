from typing import List
import torch.nn as nn
import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor
from einops.einops import rearrange
from comer.utils.utils import Hypothesis
from .dct_encoder import DCTNet
from .decoder import Decoder
from .encoder import Encoder


class CoMER(pl.LightningModule):
    def __init__(
            self,
            d_model: int,
            growth_rate: int,
            num_layers: int,
            num_mlp_layers: int,
            nhead: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout: float,
            dc: int,
            cross_coverage: bool,
            self_coverage: bool,
    ):
        super().__init__()
        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.dct_encoder = DCTNet(
            d_model=d_model, num_mlp_layers=num_mlp_layers)
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )
        self.FAB = FAB(in_dim=d_model)

    def forward(
            self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor, img_dct: FloatTensor
    ) -> FloatTensor:
        """run img and bi-tgt
        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]
        img_dct: FloatTensor
            [b, 1, h', w']
        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, h, w, d]
        dct_feature = self.dct_encoder(img_dct, mask)
        feature = feature + self.FAB(feature, dct_feature)
        feature = torch.cat((feature, feature), dim=0)
        mask = torch.cat((mask, mask), dim=0)
        out = self.decoder(feature, mask, tgt)

        return out

    def beam_search(
            self,
            img: FloatTensor,
            img_mask: LongTensor,
            img_dct: FloatTensor,
            beam_size: int,
            max_len: int,
            alpha: float,
            early_stopping: bool,
            temperature: float,
            **kwargs,
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        img_dct: FloatTensor
            [b, 1, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, h, w, d]
        dct_feature = self.dct_encoder(img_dct, mask)
        feature = feature + self.FAB(feature, dct_feature)
        return self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature)

class FAB(nn.Module):
    """
    Fusion and Alignment Block (FAB) in MFH
    """
    def __init__(self, in_dim):
        super(FAB, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_dim * 2,
            out_channels=2,
            kernel_size=3,
            padding=1
        )
        self.v_x = nn.Parameter(torch.randn((1, in_dim, 1, 1)), requires_grad=True)
        self.v_y = nn.Parameter(torch.randn((1, in_dim, 1, 1)), requires_grad=True)

    def forward(self, x, y):
        x = rearrange(x, "b h w d -> b d h w")
        y = rearrange(y, "b h w d -> b d h w")
        attmap = self.conv(torch.cat((x, y), 1))
        attmap = torch.sigmoid(attmap)
        x = attmap[:, 0:1, :, :] * x * self.v_x
        y = attmap[:, 1:, :, :] * y * self.v_y
        out = x + y
        out = rearrange(out, "b d h w  -> b h w d")
        return out
