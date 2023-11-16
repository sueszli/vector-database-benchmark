"""
Notebook styled examples
========================

The gallery is capable of transforming Python files into reStructuredText files
with a notebook structure. For this to be used you need to respect some syntax
rules.

It makes a lot of sense to contrast this output rst file with the
:download:`original Python script <plot_notebook.py>` to get better feeling of
the necessary file structure.

Anything before the Python script docstring is ignored by sphinx-gallery and
will not appear in the rst file, nor will it be executed.
This Python docstring requires an reStructuredText title to name the file and
correctly build the reference links.

Once you close the docstring you would be writing Python code. This code gets
executed by sphinx gallery shows the plots and attaches the generating code.
Nevertheless you can break your code into blocks and give the rendered file
a notebook style. In this case you have to include a code comment breaker
a line of at least 20 hashes and then every comment start with the a new hash.

As in this example we start by first writing this module
style docstring, then for the first code block we write the example file author
and script license continued by the import modules instructions.

Original script from:
https://sphinx-gallery.readthedocs.io/en/latest/tutorials/plot_notebook.html
"""
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-np.pi, np.pi, 300)
(xx, yy) = np.meshgrid(x, x)
z = np.cos(xx) + np.cos(yy)
plt.figure()
plt.imshow(z)
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.figure()
plt.imshow(z, cmap=plt.cm.get_cmap('hot'))
plt.figure()
plt.imshow(z, cmap=plt.cm.get_cmap('Spectral'), interpolation='none')

def dummy():
    if False:
        while True:
            i = 10
    "Dummy function to make sure docstrings don't get rendered as text"
    pass
string = "\nTriple-quoted string which tries to break parser but doesn't.\n"
print('Some output from Python')
plt.show()