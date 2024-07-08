import torch
import torch.nn as nn
from scipy.fftpack import dct
from torch import FloatTensor


class DCT(nn.Module):
    def __init__(self, block_size: int, freq_remove_num: int):
        super(DCT, self).__init__()
        self.block_size = block_size
        self.freq_remove_num = freq_remove_num

    def forward(self, x: FloatTensor) -> FloatTensor:
        """ Perform DCT on input images

        @param x: FloatTensor [1, h, w]
        @return: FloatTensor [1, h_padding, w_padding]
        """
        x = self.padding(x)
        x = (x * 255).byte()  # Convert binary image to grayscale image
        x_block = x.reshape(1, x.size(1) // self.block_size, self.block_size,
                            x.size(2) // self.block_size, self.block_size).permute(0, 1, 3, 2, 4)  # [1,h,w,8,8]
        # Perform 2D DCT on the last two dimensions
        # Perform DCT on each small block
        x_dct_block = torch.stack(
            [torch.tensor(dct(dct(block, axis=1, norm='ortho'), axis=2, norm='ortho')) for block in x_block.numpy()])
        x_high_freq_block = self.retain_high_freq(x_dct_block)
        x_high_freq_block = x_high_freq_block.permute(0, 1, 3, 2, 4).contiguous()
        x_dct = x_high_freq_block.view(1, x.size(1), x.size(2))
        return x_dct

    def retain_high_freq(self, dct_blocks):
        # Get the shape of each block
        _, h, w, block_size0, block_size1 = dct_blocks.shape
        # Select the range of low frequency components to remove
        remove_range_i = slice(0, self.freq_remove_num)
        remove_range_j = slice(0, self.freq_remove_num)
        # Set unwanted frequencies to zero
        for i in range(h):
            for j in range(w):
                dct_blocks[:, i, j, :, :][:, remove_range_i, :] = 0
                dct_blocks[:, i, j, :, :][:, :, remove_range_j] = 0
        return dct_blocks

    def padding(self, x):
        # Get the height and width of the input tensor
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
        # Calculate the amount of fill required
        h1 = (16 * h - x.shape[1]) // 2  # Fill the image to the corresponding value so that the shape can match the original feature
        h2 = (16 * h - x.shape[1]) - h1  # This filling is to make the original image in the middle
        w1 = (16 * w - x.shape[2]) // 2
        w2 = (16 * w - x.shape[2]) - w1
        pad = nn.ZeroPad2d(padding=(w1, w2, h1, h2))
        y = pad(x)
        return y
