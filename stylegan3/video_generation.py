import dnnlib
import legacy
import numpy as np
import torch
from typing import List, Tuple

import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm


def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

#----------------------------------------------------------------------------

def gen_interp_video(G, mp4: str, seeds, shuffle_seed=None, w_frames=60*4, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, device=torch.device('cuda'), **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if len(seeds) % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(seeds) // (grid_w*grid_h)

    all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
    for idx in range(num_keyframes*grid_h*grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
    ws = G.mapping(z=zs, c=None, truncation_psi=psi)
    _ = G.synthesis(ws[:1]) # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    # Render video.
    video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
                img = G.synthesis(ws=w.unsqueeze(0), noise_mode='const')[0]
                imgs.append(img)
        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()


class StyleGAN3VideoGenerationPipeline:
    def __init__(self, pkl_filepath_or_url='wikiart-1024-stylegan3-t-17.2Mimg.pkl', device: str = None):

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        with dnnlib.util.open_url(pkl_filepath_or_url) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)

    def __call__(
        self,
        seeds: List[int],
        outfile: str = 'out.mp4',
        truncation_psi: float = 1.0,
        grid_dims: Tuple[int, int] = (1, 1),
        w_frames: int = 60 * 4,
        bitrate: str = '12M',
    ):

        gen_interp_video(
            G=self.G,
            mp4=outfile,
            bitrate=bitrate,
            grid_dims=grid_dims,
            num_keyframes=None,
            w_frames=w_frames,
            seeds=seeds,
            psi=truncation_psi,
            device=self.device
        )
        return 'out.mp4'
