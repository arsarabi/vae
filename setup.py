#!/usr/bin/env python

import setuptools

with open('requirements.txt') as f:
    install_requires = f.read()

setuptools.setup(
    name='vae',
    version='0.1.0',
    packages=setuptools.find_packages(),
    install_requires=install_requires
)
