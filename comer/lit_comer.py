import zipfile
from typing import List

import pytorch_lightning as pl
import torch.optim as optim
from torch import FloatTensor, LongTensor
import torch
from comer.datamodule import Batch, vocab
from comer.model.comer import CoMER
from comer.utils.utils import (ExpRateRecorder, Hypothesis, ce_loss,
                               to_bi_tgt_out)


class LitCoMER(pl.LightningModule):
    def __init__(
            self,
            d_model: int,
            # encoder
            growth_rate: int,
            num_layers: int,
            num_mlp_layers: int,
            # decoder
            nhead: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout: float,
            dc: int,
            cross_coverage: bool,
            self_coverage: bool,
            # beam search
            beam_size: int,
            max_len: int,
            alpha: float,
            early_stopping: bool,
            temperature: float,
            # training
            learning_rate: float,
            patience: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.comer_model = CoMER(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            num_mlp_layers=num_mlp_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        self.exprate_recorder = ExpRateRecorder()

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
        return self.comer_model(img, img_mask, tgt, img_dct)

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt, batch.img_dct)

        loss = ce_loss(out_hat, out)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt, batch.img_dct)

        loss = ce_loss(out_hat, out)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        hyps = self.approximate_joint_search(batch.imgs, batch.mask, batch.img_dct)

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Batch, _):
        hyps = self.approximate_joint_search(batch.imgs, batch.mask, batch.img_dct)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        return batch.img_bases, [vocab.indices2label(h.seq) for h in hyps]

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")

    def approximate_joint_search(
            self, img: FloatTensor, mask: LongTensor, img_dct: FloatTensor
    ) -> List[Hypothesis]:
        return self.comer_model.beam_search(img, mask, img_dct, **self.hparams)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.25,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
