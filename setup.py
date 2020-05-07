"""Pylot: A light and fast aircraft flight simulator."""

from setuptools import setup
import os
import sys

setup(name = 'Pylot',
    version = '1.1',
    description = "Pylot: A light and fast aircraft flight simulator.",
    url = 'https://github.com/usuaero/Pylot',
    author = 'usuaero',
    author_email = 'doug.hunsaker@usu.edu',
    install_requires = ['pygame', 'pyrr', 'PyOpenGL_accelerate', 'PyOpenGL', 'numpy', 'scipy', 'pytest', 'matplotlib', 'pynput', 'inputs'],
    python_requires ='>=3.6.0',
    include_package_data=True,
    license = 'MIT',
    packages = ['pylot'],
    zip_safe = False)