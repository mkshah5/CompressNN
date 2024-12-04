from setuptools import setup, find_packages

import os

# build the extension module
setup(
    name='compressnn',
    author = 'Milan Shah',
    author_email='mkshah5@ncsu.edu',
    version='0.0.3',
    packages=find_packages(),
    install_requires=[
        'torch'
    ]
)
