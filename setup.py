from distutils.core import setup, Extension
from setuptools import find_packages
from distutils.command.build_ext import build_ext

msvc_args = ['/openmp', '/Ox', '/fp:fast', '/favor:INTEL64', '/Og']
unix_args = ['-std=c++11', '-fopenmp', '-O3', '-ffast-math', '-march=native']

copt = {'msvc': msvc_args,
        'mingw32':  unix_args,
        'mingw64': unix_args,
        'unix': unix_args}


class build_ext_subclass(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        print("COMPILER ", c)
        if c in copt:
            for e in self.extensions:
                e.extra_compile_args = copt[c]
        else:
            print("WARNING: UNKNOWN COMPILER")
        build_ext.build_extensions(self)


module1 = Extension('tagextractor.native',
                    include_dirs=['libs/eigen', 'libs/pybind11/include'],
                    sources=['tagextractor/native/tagextractor.cpp'])

setup(name='tagextractor',
      version='0.1',
      description='This package for tfidf tags extraction',
      packages=find_packages(exclude=["libs"]),
      ext_modules=[module1],
      cmdclass={'build_ext': build_ext_subclass})
