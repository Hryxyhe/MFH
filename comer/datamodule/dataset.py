import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset
from .dct_transform import DCT
from .transforms import ScaleAugmentation, ScaleToLimitRange
import copy

K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024


class CROHMEDataset(Dataset):
    def __init__(self, ds, is_train: bool, scale_aug: bool, freq_remove_num: int) -> None:
        super().__init__()
        self.ds = ds

        trans_list = []
        if is_train and scale_aug:
            trans_list.append(ScaleAugmentation(K_MIN, K_MAX))

        trans_list += [
            ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI),
            tr.ToTensor(),
        ]
        self.transform = tr.Compose(trans_list)
        self.dct = DCT(block_size=8, freq_remove_num=freq_remove_num)
    def __getitem__(self, idx):
        fname, img, caption = self.ds[idx]
        img = [self.transform(im) for im in img]
        img_dct = copy.deepcopy(img)  # gain the copy of input image for DCT
        img_dct = [self.dct(x) for x in img_dct]
        return fname, img, caption, img_dct

    def __len__(self):
        return len(self.ds)


