import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.absolute().as_posix())
from .image_generation import StyleGAN3ImageGenerationPipeline
from .video_generation import StyleGAN3VideoGenerationPipeline
sys.path.pop()
