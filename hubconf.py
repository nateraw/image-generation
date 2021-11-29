import gdown

from huggingface_stylegan3 import StyleGAN3ImageGenerationPipeline

model_map = {
    'art': 'https://drive.google.com/uc?id=18MOpwTMJsl_Z17q-wQVnaRLCUFZYSNkj',
    'landscapes': 'https://drive.google.com/uc?id=14UGDDOusZ9TMb-pOrF0PAjMGVWLSAii1'
}

def styleganv3(pretrained: str = 'art'):
    if pretrained.startswith('https://'):
        url = pretrained
    elif pretrained in model_map:
        url = model_map.get(pretrained, None)
    elif '/' in pretrained and len(pretrained.split('/')) == 2:
        url = 'https://hf.co/{pretrained}/resolve/main/model.pkl'
    else:
        raise ValueError(f'Model {pretrained} not found')
    return StyleGAN3ImageGenerationPipeline(url)
