from distutils.core import setup, Extension
from setuptools import find_packages

module1 = Extension('tagextractor.native',
                    include_dirs = ['libs/eigen', 'libs/pybind11/include'],
                    sources = ['tagextractor/native/tagextractor.cpp'],
                    extra_compile_args=['-std=c++11'])

setup (name = 'tagextractor',
       version = '0.1',
       description = 'This package for tfidf tags extraction',
       packages=find_packages(exclude=["libs"]),
       ext_modules = [module1])
