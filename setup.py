from os import path

import setuptools
from setuptools import setup

setup(
    name="tjc-gym",
    version="0.0.1",
    description="OpenAI Gym environment of Traffic Junction with continuous action space",
    long_description_content_type='text/markdown',
    long_description=open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8').read(),
    url='https://github.com/jakobdybdahl/tjc-gym',
    author="Jakob Dybdahl, Rasmus Thorsen",
    author_email="dybdahl@smukand.dk",
    license='MIT License',
    packages=setuptools.find_packages(),
    install_requires=["gym==0.21.0", "numpy>=1.21.4", "pyglet>=1.5.21"],
    classifiers=[
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9'
      ],
)
