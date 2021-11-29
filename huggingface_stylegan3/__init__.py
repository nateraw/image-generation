import sys
from pathlib import Path

sys.path.append(Path(__file__).parent.absolute().as_posix())
from .pipeline import StyleGAN3ImageGenerationPipeline

sys.path.pop()
