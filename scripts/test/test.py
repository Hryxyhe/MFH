import os
import sys

# sys.path.append('your absolute path-to-this project')
sys.path.append("/home/ipad_3d/yhx_wqw/MFH/")
import typer
from comer.datamodule.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER
from pytorch_lightning import Trainer, seed_everything

seed_everything(7)


# def main(version: str, test_year: str):
def main(version: str):
    test_years = ['2014', '2016', '2019']
    # generate output latex in result.zip
    for test_year in test_years:
        ckp_folder = os.path.join("lightning_logs", f"version_{version}", "checkpoints")
        fnames = os.listdir(ckp_folder)
        assert len(fnames) == 1
        ckp_path = os.path.join(ckp_folder, fnames[0])
        print(f"Test with fname: {fnames[0]}")

        trainer = Trainer(logger=False, gpus=1)

        dm = CROHMEDatamodule(test_year=test_year, eval_batch_size=4)

        model = LitCoMER.load_from_checkpoint(ckp_path)

        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    typer.run(main)
