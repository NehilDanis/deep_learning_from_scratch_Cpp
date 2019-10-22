import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ['-std=c++11']

ext_modules = [
    Extension(
        'dl',
        ['Activation.cpp', 'AddNode.cpp', 'ComputationalNode.cpp', 'Connector.cpp', 'Convolution.cpp', 'DotProduct.cpp', 'Maxpooling.cpp',
         'Network.cpp', 'Dropout.cpp', 'wrap.cpp'],
        include_dirs=['../libs/include'],
        language='c++',
        extra_compile_args = cpp_args,
    ),
]

setup(
    name='dl',
    version='0.0.1',
    author='Nehil',
    author_email='nehil.danis@tum.de',
    description='Example',
    ext_modules=ext_modules,
)