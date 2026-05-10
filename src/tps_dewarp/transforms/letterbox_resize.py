import torch
import torchvision.transforms.functional as F

class LetterboxResize:
    def __init__(self, size=256, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        # img: Tensor [C,H,W]

        _, h, w = img.shape

        scale = min(self.size / h, self.size / w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        img = F.resize(img, [new_h, new_w])

        pad_h = self.size - new_h
        pad_w = self.size - new_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        img = F.pad(
            img,
            [pad_left, pad_top, pad_right, pad_bottom],
            fill=self.fill
        )

        return img
