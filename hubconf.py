import gdown

from huggingface_stylegan3 import StyleGAN3ImageGenerationPipeline

stylegan_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/'

nvidia_models = [
    'stylegan3-t-afhqv2-512x512.pkl',
    'stylegan3-r-afhqv2-512x512.pkl'
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
    'art': 'https://drive.google.com/uc?id=18MOpwTMJsl_Z17q-wQVnaRLCUFZYSNkj',
    'landscapes': 'https://drive.google.com/uc?id=14UGDDOusZ9TMb-pOrF0PAjMGVWLSAii1',
    **{m: f'{stylegan_url}{m}' for m in nvidia_models},
}

def model_map():
    return _model_map

def styleganv3(pretrained: str = 'art'):
    if pretrained.startswith('https://'):
        url = pretrained
    elif pretrained in _model_map:
        url = _model_map.get(pretrained, None)
    elif '/' in pretrained and len(pretrained.split('/')) == 2:
        url = 'https://hf.co/{pretrained}/resolve/main/model.pkl'
    else:
        raise ValueError(f'Model {pretrained} not found')
    return StyleGAN3ImageGenerationPipeline(url)
