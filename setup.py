from setuptools import setup, find_packages

setup(
    name="msaclip",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'transformers>=4.30.0',
        'Pillow>=8.0.0',
        'scikit-learn>=0.24.0',
        'numpy>=1.21.0',
        'tqdm>=4.65.0',
        'pandas>=1.5.0',
    ]
) 