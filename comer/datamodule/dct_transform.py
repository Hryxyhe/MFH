import torch
import torch.nn as nn
from scipy.fftpack import dct, idct


class DCT(nn.Module):
    def __init__(self, block_size: int, freq_ratio: int):
        super(DCT, self).__init__()
        self.block_size = block_size
        self.freq_ratio = freq_ratio

    def forward(self, x):
        x = self.padding(x)  # x: 1, h, w
        x = (x * 255).byte()
        x_block = x.reshape(1, x.size(1) // self.block_size, self.block_size,
                            x.size(2) // self.block_size, self.block_size).permute(0, 1, 3, 2, 4)  # [1,h,w,8,8]
        x_dct_block = torch.stack(
            [torch.tensor(dct(dct(block, axis=1, norm='ortho'), axis=2, norm='ortho')) for block in x_block.numpy()])
        x_idct_block = self.retain_high_freq(x_dct_block)
        x_idct_block = x_idct_block.permute(0, 1, 3, 2, 4).contiguous()
        x_idct = x_idct_block.view(1, x.size(1), x.size(2))

        return x_idct

    def padding(self, x):
        _, h, w = x.size()
        for _ in range(4):
            if h % 2 == 0:
                h = h // 2
            else:
                h = (h + 1) // 2
            if w % 2 == 0:
                w = w // 2
            else:
                w = (w + 1) // 2
        h1 = (16 * h - x.shape[1]) // 2
        h2 = (16 * h - x.shape[1]) - h1
        w1 = (16 * w - x.shape[2]) // 2
        w2 = (16 * w - x.shape[2]) - w1
        pad = nn.ZeroPad2d(padding=(w1, w2, h1, h2))
        y = pad(x)
        return y

    def retain_high_freq(self, dct_blocks):
        dct_blocks = dct_blocks
        _, h, w, block_size0, block_size1 = dct_blocks.shape
        keep_range_i = slice(0, self.freq_ratio)  # 0 to 5 (inclusive)
        keep_range_j = slice(0, self.freq_ratio)  # 0 to 5 (inclusive)
        for i in range(h):
            for j in range(w):
                dct_blocks[:, i, j, :, :][:, keep_range_i, :] = 0
                dct_blocks[:, i, j, :, :][:, :, keep_range_j] = 0
        return dct_blocks
