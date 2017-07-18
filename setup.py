import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "deepstacks",
    version = "0.0.2",
    author = "Guo Xuesong",
    author_email = "zenothing@hotmail.com",
    description = ("A build_network() for Lasagne and noen."
        "Define your network model in a datacheet with stack machine mechanisms."
        "Support reuse part of model as function, and share parameters."
        ),
    license = "MIT",
    keywords = "example documentation tutorial",
    url = "http://packages.python.org/deepstacks",
    packages=['deepstacks', 'tests'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
