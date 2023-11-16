"""
Handlers for IPythonDirective's @doctest pseudo-decorator.

The Sphinx extension that provides support for embedded IPython code provides
a pseudo-decorator @doctest, which treats the input/output block as a
doctest, raising a RuntimeError during doc generation if the actual output
(after running the input) does not match the expected output.

An example usage is:

.. code-block:: rst

   .. ipython::

        In [1]: x = 1

        @doctest
        In [2]: x + 2
        Out[3]: 3

One can also provide arguments to the decorator. The first argument should be
the name of a custom handler. The specification of any other arguments is
determined by the handler. For example,

.. code-block:: rst

      .. ipython::

         @doctest float
         In [154]: 0.1 + 0.2
         Out[154]: 0.3

allows the actual output ``0.30000000000000004`` to match the expected output
due to a comparison with `np.allclose`.

This module contains handlers for the @doctest pseudo-decorator. Handlers
should have the following function signature::

    handler(sphinx_shell, args, input_lines, found, submitted)

where `sphinx_shell` is the embedded Sphinx shell, `args` contains the list
of arguments that follow: '@doctest handler_name', `input_lines` contains
a list of the lines relevant to the current doctest, `found` is a string
containing the output from the IPython shell, and `submitted` is a string
containing the expected output from the IPython shell.

Handlers must be registered in the `doctests` dict at the end of this module.

"""

def str_to_array(s):
    if False:
        print('Hello World!')
    '\n    Simplistic converter of strings from repr to float NumPy arrays.\n\n    If the repr representation has ellipsis in it, then this will fail.\n\n    Parameters\n    ----------\n    s : str\n        The repr version of a NumPy array.\n\n    Examples\n    --------\n    >>> s = "array([ 0.3,  inf,  nan])"\n    >>> a = str_to_array(s)\n\n    '
    import numpy as np
    from numpy import inf, nan
    if s.startswith(u'array'):
        s = s[6:-1]
    if s.startswith(u'['):
        a = np.array(eval(s), dtype=float)
    else:
        a = np.atleast_1d(float(s))
    return a

def float_doctest(sphinx_shell, args, input_lines, found, submitted):
    if False:
        for i in range(10):
            print('nop')
    '\n    Doctest which allow the submitted output to vary slightly from the input.\n\n    Here is how it might appear in an rst file:\n\n    .. code-block:: rst\n\n       .. ipython::\n\n          @doctest float\n          In [1]: 0.1 + 0.2\n          Out[1]: 0.3\n\n    '
    import numpy as np
    if len(args) == 2:
        rtol = 1e-05
        atol = 1e-08
    else:
        try:
            rtol = float(args[2])
            atol = float(args[3])
        except IndexError as e:
            e = 'Both `rtol` and `atol` must be specified if either are specified: {0}'.format(args)
            raise IndexError(e) from e
    try:
        submitted = str_to_array(submitted)
        found = str_to_array(found)
    except:
        error = True
    else:
        found_isnan = np.isnan(found)
        submitted_isnan = np.isnan(submitted)
        error = not np.allclose(found_isnan, submitted_isnan)
        error |= not np.allclose(found[~found_isnan], submitted[~submitted_isnan], rtol=rtol, atol=atol)
    TAB = ' ' * 4
    directive = sphinx_shell.directive
    if directive is None:
        source = 'Unavailable'
        content = 'Unavailable'
    else:
        source = directive.state.document.current_source
        content = '\n'.join([TAB + line for line in directive.content])
    if error:
        e = 'doctest float comparison failure\n\nDocument source: {0}\n\nRaw content: \n{1}\n\nOn input line(s):\n{TAB}{2}\n\nwe found output:\n{TAB}{3}\n\ninstead of the expected:\n{TAB}{4}\n\n'
        e = e.format(source, content, '\n'.join(input_lines), repr(found), repr(submitted), TAB=TAB)
        raise RuntimeError(e)
doctests = {'float': float_doctest}