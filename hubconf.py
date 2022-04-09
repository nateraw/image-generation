from stylegan3 import StyleGAN3ImageGenerationPipeline, StyleGAN3VideoGenerationPipeline


stylegan_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/'

nvidia_models = [
    "stylegan3-t-ffhq-1024x1024.pkl",
    "stylegan3-t-ffhqu-1024x1024.pkl",
    "stylegan3-t-ffhqu-256x256.pkl",
    "stylegan3-r-ffhq-1024x1024.pkl",
    "stylegan3-r-ffhqu-1024x1024.pkl",
    "stylegan3-r-ffhqu-256x256.pkl",
    "stylegan3-t-metfaces-1024x1024.pkl",
    "stylegan3-t-metfacesu-1024x1024.pkl",
    "stylegan3-r-metfaces-1024x1024.pkl",
    "stylegan3-r-metfacesu-1024x1024.pkl",
    "stylegan3-t-afhqv2-512x512.pkl",
    "stylegan3-r-afhqv2-512x512.pkl",
]

_model_map = {
    'wikiart-1024': 'https://archive.org/download/wikiart-1024-stylegan3-t-17.2Mimg/wikiart-1024-stylegan3-t-17.2Mimg.pkl',
    'landscapes-256': 'https://archive.org/download/lhq-256-stylegan3-t-25Mimg/lhq-256-stylegan3-t-25Mimg.pkl',
    'mechanical-devices-256': 'https://www.dropbox.com/s/v2oie53cz62ozvu/network-snapshot-000029.pkl?dl=1',
    'vivid-flowers-256': 'https://www.dropbox.com/s/o33lhgnk91hstvx/network-snapshot-000069.pkl?dl=1',
    'alien-with-sunglasses-256': 'https://www.dropbox.com/s/gur14k0e7kspguy/network-snapshot-000038.pkl?dl=1',
    'forest-daemons-256': 'https://www.dropbox.com/s/26muctr2eq4br6l/network-snapshot-000018.pkl?dl=1',
    'scifi-city-256': 'https://www.dropbox.com/s/1kfsmlct4mriphc/network-snapshot-000210.pkl?dl=1',
    'scifi-spaceship-256': 'https://www.dropbox.com/s/02br3mjkma1hubc/network-snapshot-000162.pkl?dl=1',
    'yellow-comic-alien-256': 'https://www.dropbox.com/s/g14lsdbohuk2oco/network-snapshot-000019.pkl?dl=1',
    'yellow-comic-alien-512': 'https://www.dropbox.com/s/yzraojzmg2kybjx/network-snapshot-000236.pkl?dl=1',
    'mechanical-landscape-256': 'https://www.dropbox.com/s/5oar443k7iw3qvp/network-snapshot-000006.pkl?dl=1',
    **{m: f'{stylegan_url}{m}' for m in nvidia_models},
}

def model_map():
    return _model_map

def styleganv3(pretrained: str = 'art', videos: bool = False):
    if pretrained.startswith('https://'):
        url = pretrained
    elif pretrained in _model_map:
        url = _model_map.get(pretrained, None)
    elif '/' in pretrained and len(pretrained.split('/')) == 2:
        url = 'https://hf.co/{pretrained}/resolve/main/model.pkl'
    else:
        raise ValueError(f'Model {pretrained} not found')
    return StyleGAN3ImageGenerationPipeline(url) if not videos else StyleGAN3VideoGenerationPipeline(url)

