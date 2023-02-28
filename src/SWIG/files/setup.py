from distutils.core import setup, Extension
import numpy
import os

# Define the extension module
simple_module = Extension('_simple',
                         sources=['simple_wrap.c', 'simple.c'],
                         include_dirs=[numpy.get_include(),'.']
                         )

# Define the setup parameters
setup(name='simple',
      version='1.0',
      author='Stefan',
      description='SAXPY using SWIG and C',
      ext_modules=[simple_module],
      py_modules=["simple"],
      )
