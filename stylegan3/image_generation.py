import pickle
from typing import List, Optional, Tuple, Union

import dnnlib
import legacy
import numpy as np
import PIL.Image
import torch


def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


class StyleGAN3ImageGenerationPipeline:
    def __init__(self, pkl_filepath_or_url='wikiart-1024-stylegan3-t-17.2Mimg.pkl', device: str = None):

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        with dnnlib.util.open_url(pkl_filepath_or_url) as f:
#             self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
            self.G = pickle.load(f)['G_ema'].to(self.device)

    def __call__(
        self,
        seed: int = 42,
        truncation_psi: float = 1.0,
        class_idx: Optional[int] = None,
        noise_mode: str = 'const',
        translate: Tuple[int] = (0, 0),
        rotate: float = 0.0,
    ):

        # Labels.
        self.label = torch.zeros([1, self.G.c_dim], device=self.device)
        if self.G.c_dim != 0:
            if class_idx is None:
                raise RuntimeError('Must specify class label with --class when using a conditional network')
            self.label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print('warn: --class=lbl ignored when running on an unconditional network')

        # Generate images.
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device)

        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(self.G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            self.G.synthesis.input.transform.copy_(torch.from_numpy(m))

        img = self.G(z, self.label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        return img
