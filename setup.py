from setuptools import setup, find_packages

import io
from os import path

PACKAGE_NAME = 'deepgaze_pytorch'
VERSION = '0.2.0'
DESCRIPTION = 'Python pytorch implementation of the different DeepGaze models'
AUTHOR = 'Matthias Kümmerer'
EMAIL = 'matthias.kuemmerer@bethgelab.org'
URL = "https://github.com/matthiask/deepgaze"

try:
    this_directory = path.abspath(path.dirname(__file__))
    with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except IOError:
    long_description = ''

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        # "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(),
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    # license='MIT',
    install_requires=[
        'boltons',
        'numpy',
        'torch',
        'torchvision',
        'setuptools',
    ],
    include_package_data=True,
    package_data={'deepgaze_pytorch': ['*.yaml']},
)
