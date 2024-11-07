from setuptools import setup, find_packages

import os

# build the extension module
setup(
    name='compressnn',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'torch'
    ]
)
