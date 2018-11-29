# -*- coding: utf-8 -*-
import os
from distutils.core import setup
import codecs

# shamelessly copied from VoroPy
def read(fname):
    return codecs.open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()

setup(name='pseudopy',
      packages=['pseudopy'],
      version='1.2.5',
      description='Compute and visualize pseudospectra of'
                  + ' matrices (like eigtool)',
      long_description=read('README.md'),
      author='André Gaul',
      author_email='gaul@web-yard.de',
      url='https://github.com/andrenarchy/pseudopy',
      install_requires=['matplotlib>=2.0', 'numpy>=1.7',
                'scipy>=0.12', 'shapely>=1.2'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Mathematics'
          ],
      )
