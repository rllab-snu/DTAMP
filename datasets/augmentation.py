import numpy as np
import torch
from torchvision.transforms.functional import affine
from torchvision.transforms import ColorJitter
from copy import deepcopy


class Transform:
    def __init__(self, max_translate=6, brightness=0.1, contrast=0.1, hue=0.02):
        self.color_jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            hue=hue
        )
        self.max_translate = max_translate

    def augment(self, image, eval=False):
        if not eval:
            translate = list(np.random.randint(-self.max_translate, self.max_translate + 1, size=2))
            image = affine(image, angle=0, translate=translate, scale=1, shear=[0])
            image = self.color_jitter(image)
        image = 2. * image / 255. - 1.
        return image

    def __call__(self, obs, eval=False):
        raw_obs = self.augment(obs, eval=True)
        if len(obs.shape) == 4:
            obs = self.augment(obs, eval)
        elif len(obs.shape) == 5:
            obs = torch.stack(
                [self.augment(x, eval) for x in obs.transpose(0, 1)], dim=1
            )
        return obs, raw_obs
