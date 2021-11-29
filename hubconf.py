import gdown

from huggingface_stylegan3 import StyleGAN3ImageGenerationPipeline

model_map = {
    'art': 'https://drive.google.com/uc?id=18MOpwTMJsl_Z17q-wQVnaRLCUFZYSNkj',
    'landscapes': 'https://drive.google.com/uc?id=14UGDDOusZ9TMb-pOrF0PAjMGVWLSAii1'
}

def styleganv3_inference(model: str = 'art'):
    if model.startswith('http'):
        url = model
    elif model in model_map:
        url = model_map.get(model, None)
    else:
        raise ValueError(f'Model {model} not found')
    return StyleGAN3ImageGenerationPipeline(url)
