from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name='huggingface-stylegan3',
    packages=find_packages(),
    version='0.0.1',
    # license=, # TODO - codebase heavily draws from stylegan3. need to issue license correctly.
    description='Pipelines',
    author='Nathan Raw',
    author_email='naterawdata@gmail.com',
    url='https://github.com/nateraw/huggingface-stylegan3',
    install_requires=requirements,
)