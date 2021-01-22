import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import shutil
import numpy

print('Build extension modules...')
print('==============================================')

ext_modules = [Extension('core',
		['src/core/core.pyx', 
		'src/core/CTF.cpp'],
		language='c++',
		include_dirs=[numpy.get_include()],
		extra_compile_args=["-O2"]
	       )]

setup(
	name = 'Extended Cython module',
	cmdclass = {'build_ext': build_ext},
	ext_modules = ext_modules
)

shutil.move('core.cpython-37m-x86_64-linux-gnu.so', 'src/core.so')
print('==============================================')
print('Build done.\n')
