"""Packaging for SyntaxNet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import setuptools
import setuptools.dist
include_tensorflow = os.path.isdir('tensorflow')
source_roots = ['dragnn', 'syntaxnet'] + (['tensorflow'] if include_tensorflow else [])

def data_files():
    if False:
        i = 10
        return i + 15
    'Return all non-Python files in the source directories.'
    for root in source_roots:
        for (path, _, files) in os.walk(root):
            for filename in files:
                if not (filename.endswith('.py') or filename.endswith('.pyc')):
                    yield os.path.join(path, filename)

class BinaryDistribution(setuptools.dist.Distribution):
    """Copied from TensorFlow's setup script: sets has_ext_modules=True.

  Distributions of SyntaxNet include shared object files, which are not
  cross-platform.
  """

    def has_ext_modules(self):
        if False:
            i = 10
            return i + 15
        return True
with open('MANIFEST.in', 'w') as f:
    f.write(''.join(('include {}\n'.format(filename) for filename in data_files())))
setuptools.setup(name='syntaxnet_with_tensorflow' if include_tensorflow else 'syntaxnet', version='0.2', description='SyntaxNet: Neural Models of Syntax', long_description='', url='https://github.com/tensorflow/models/tree/master/syntaxnet', author='Google Inc.', author_email='opensource@google.com', packages=setuptools.find_packages(), install_requires=([] if include_tensorflow else ['tensorflow']) + ['pygraphviz'], include_package_data=True, zip_safe=False, distclass=BinaryDistribution, classifiers=['Intended Audience :: Developers', 'Intended Audience :: Education', 'Intended Audience :: Science/Research', 'License :: OSI Approved :: Apache Software License', 'Programming Language :: Python :: 2.7', 'Topic :: Scientific/Engineering :: Mathematics'], license='Apache 2.0', keywords='syntaxnet machine learning')