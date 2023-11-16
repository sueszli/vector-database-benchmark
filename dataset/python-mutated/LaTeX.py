"""Dependency scanner for LaTeX code."""
import os.path
import re
import SCons.Node.FS
import SCons.Util
import SCons.Warnings
from . import ScannerBase, FindPathDirs
TexGraphics = ['.eps', '.ps']
LatexGraphics = ['.png', '.jpg', '.gif', '.tif']

class _Null:
    pass
_null = _Null

def modify_env_var(env, var, abspath):
    if False:
        return 10
    try:
        save = env['ENV'][var]
    except KeyError:
        save = _null
    env.PrependENVPath(var, abspath)
    try:
        if SCons.Util.is_List(env[var]):
            env.PrependENVPath(var, [os.path.abspath(str(p)) for p in env[var]])
        else:
            env.PrependENVPath(var, [os.path.abspath(p) for p in str(env[var]).split(os.pathsep)])
    except KeyError:
        pass
    if SCons.Util.is_List(env['ENV'][var]):
        env['ENV'][var] = os.pathsep.join(env['ENV'][var])
    env['ENV'][var] = env['ENV'][var] + os.pathsep
    return save

class FindENVPathDirs:
    """
    A class to bind a specific E{*}PATH variable name to a function that
    will return all of the E{*}path directories.
    """

    def __init__(self, variable):
        if False:
            return 10
        self.variable = variable

    def __call__(self, env, dir=None, target=None, source=None, argument=None):
        if False:
            print('Hello World!')
        import SCons.PathList
        try:
            path = env['ENV'][self.variable]
        except KeyError:
            return ()
        dir = dir or env.fs._cwd
        path = SCons.PathList.PathList(path).subst_path(env, target, source)
        return tuple(dir.Rfindalldirs(path))

def LaTeXScanner():
    if False:
        print('Hello World!')
    '\n    Return a prototype Scanner instance for scanning LaTeX source files\n    when built with latex.\n    '
    ds = LaTeX(name='LaTeXScanner', suffixes='$LATEXSUFFIXES', graphics_extensions=TexGraphics, recursive=0)
    return ds

def PDFLaTeXScanner():
    if False:
        i = 10
        return i + 15
    '\n    Return a prototype Scanner instance for scanning LaTeX source files\n    when built with pdflatex.\n    '
    ds = LaTeX(name='PDFLaTeXScanner', suffixes='$LATEXSUFFIXES', graphics_extensions=LatexGraphics, recursive=0)
    return ds

class LaTeX(ScannerBase):
    """Class for scanning LaTeX files for included files.

    Unlike most scanners, which use regular expressions that just
    return the included file name, this returns a tuple consisting
    of the keyword for the inclusion ("include", "includegraphics",
    "input", or "bibliography"), and then the file name itself.
    Based on a quick look at LaTeX documentation, it seems that we
    should append .tex suffix for the "include" keywords, append .tex if
    there is no extension for the "input" keyword, and need to add .bib
    for the "bibliography" keyword that does not accept extensions by itself.

    Finally, if there is no extension for an "includegraphics" keyword
    latex will append .ps or .eps to find the file, while pdftex may use .pdf,
    .jpg, .tif, .mps, or .png.

    The actual subset and search order may be altered by
    DeclareGraphicsExtensions command. This complication is ignored.
    The default order corresponds to experimentation with teTeX::

        $ latex --version
        pdfeTeX 3.141592-1.21a-2.2 (Web2C 7.5.4)
        kpathsea version 3.5.4

    The order is:
        ['.eps', '.ps'] for latex
        ['.png', '.pdf', '.jpg', '.tif'].

    Another difference is that the search path is determined by the type
    of the file being searched:
    env['TEXINPUTS'] for "input" and "include" keywords
    env['TEXINPUTS'] for "includegraphics" keyword
    env['TEXINPUTS'] for "lstinputlisting" keyword
    env['BIBINPUTS'] for "bibliography" keyword
    env['BSTINPUTS'] for "bibliographystyle" keyword
    env['INDEXSTYLE'] for "makeindex" keyword, no scanning support needed just allows user to set it if needed.

    FIXME: also look for the class or style in document[class|style]{}
    FIXME: also look for the argument of bibliographystyle{}
    """
    keyword_paths = {'include': 'TEXINPUTS', 'input': 'TEXINPUTS', 'includegraphics': 'TEXINPUTS', 'bibliography': 'BIBINPUTS', 'bibliographystyle': 'BSTINPUTS', 'addbibresource': 'BIBINPUTS', 'addglobalbib': 'BIBINPUTS', 'addsectionbib': 'BIBINPUTS', 'makeindex': 'INDEXSTYLE', 'usepackage': 'TEXINPUTS', 'lstinputlisting': 'TEXINPUTS'}
    env_variables = SCons.Util.unique(list(keyword_paths.values()))
    two_arg_commands = ['import', 'subimport', 'includefrom', 'subincludefrom', 'inputfrom', 'subinputfrom']

    def __init__(self, name, suffixes, graphics_extensions, *args, **kwargs):
        if False:
            print('Hello World!')
        regex = '\n            \\\\(\n                include\n              | includegraphics(?:\\s*\\[[^\\]]+\\])?\n              | lstinputlisting(?:\\[[^\\]]+\\])?\n              | input\n              | import\n              | subimport\n              | includefrom\n              | subincludefrom\n              | inputfrom\n              | subinputfrom\n              | bibliography\n              | addbibresource\n              | addglobalbib\n              | addsectionbib\n              | usepackage\n              )\n                  \\s*{([^}]*)}       # first arg\n              (?: \\s*{([^}]*)} )?    # maybe another arg\n        '
        self.cre = re.compile(regex, re.M | re.X)
        self.comment_re = re.compile('^((?:(?:\\\\%)|[^%\\n])*)(.*)$', re.M)
        self.graphics_extensions = graphics_extensions

        def _scan(node, env, path=(), self=self):
            if False:
                for i in range(10):
                    print('nop')
            node = node.rfile()
            if not node.exists():
                return []
            return self.scan_recurse(node, path)

        class FindMultiPathDirs:
            """The stock FindPathDirs function has the wrong granularity:
            it is called once per target, while we need the path that depends
            on what kind of included files is being searched. This wrapper
            hides multiple instances of FindPathDirs, one per the LaTeX path
            variable in the environment. When invoked, the function calculates
            and returns all the required paths as a dictionary (converted into
            a tuple to become hashable). Then the scan function converts it
            back and uses a dictionary of tuples rather than a single tuple
            of paths.
            """

            def __init__(self, dictionary):
                if False:
                    print('Hello World!')
                self.dictionary = {}
                for (k, n) in dictionary.items():
                    self.dictionary[k] = (FindPathDirs(n), FindENVPathDirs(n))

            def __call__(self, env, dir=None, target=None, source=None, argument=None):
                if False:
                    for i in range(10):
                        print('nop')
                di = {}
                for (k, (c, cENV)) in self.dictionary.items():
                    di[k] = (c(env, dir=None, target=None, source=None, argument=None), cENV(env, dir=None, target=None, source=None, argument=None))
                return tuple(di.items())

        class LaTeXScanCheck:
            """Skip all but LaTeX source files.

            Do not scan *.eps, *.pdf, *.jpg, etc.
            """

            def __init__(self, suffixes):
                if False:
                    i = 10
                    return i + 15
                self.suffixes = suffixes

            def __call__(self, node, env):
                if False:
                    print('Hello World!')
                current = not node.has_builder() or node.is_up_to_date()
                scannable = node.get_suffix() in env.subst_list(self.suffixes)[0]
                return scannable and current
        kwargs['function'] = _scan
        kwargs['path_function'] = FindMultiPathDirs(LaTeX.keyword_paths)
        kwargs['recursive'] = 0
        kwargs['skeys'] = suffixes
        kwargs['scan_check'] = LaTeXScanCheck(suffixes)
        kwargs['name'] = name
        super().__init__(*args, **kwargs)

    def _latex_names(self, include_type, filename):
        if False:
            return 10
        if include_type == 'input':
            (base, ext) = os.path.splitext(filename)
            if ext == '':
                return [filename + '.tex']
        if include_type in ('include', 'import', 'subimport', 'includefrom', 'subincludefrom', 'inputfrom', 'subinputfrom'):
            (base, ext) = os.path.splitext(filename)
            if ext == '':
                return [filename + '.tex']
        if include_type == 'bibliography':
            (base, ext) = os.path.splitext(filename)
            if ext == '':
                return [filename + '.bib']
        if include_type == 'usepackage':
            (base, ext) = os.path.splitext(filename)
            if ext == '':
                return [filename + '.sty']
        if include_type == 'includegraphics':
            (base, ext) = os.path.splitext(filename)
            if ext == '':
                return [filename + e for e in self.graphics_extensions]
        return [filename]

    def sort_key(self, include):
        if False:
            while True:
                i = 10
        return SCons.Node.FS._my_normcase(str(include))

    def find_include(self, include, source_dir, path):
        if False:
            for i in range(10):
                print('nop')
        (inc_type, inc_subdir, inc_filename) = include
        try:
            sub_paths = path[inc_type]
        except (IndexError, KeyError):
            sub_paths = ((), ())
        try_names = self._latex_names(inc_type, inc_filename)
        search_paths = [(source_dir,)] + list(sub_paths)
        for n in try_names:
            for search_path in search_paths:
                paths = tuple([d.Dir(inc_subdir) for d in search_path])
                i = SCons.Node.FS.find_file(n, paths)
                if i:
                    return (i, include)
        return (None, include)

    def canonical_text(self, text):
        if False:
            while True:
                i = 10
        'Standardize an input TeX-file contents.\n\n        Currently:\n          * removes comments, unwrapping comment-wrapped lines.\n        '
        out = []
        line_continues_a_comment = False
        for line in text.splitlines():
            (line, comment) = self.comment_re.findall(line)[0]
            if line_continues_a_comment:
                out[-1] = out[-1] + line.lstrip()
            else:
                out.append(line)
            line_continues_a_comment = len(comment) > 0
        return '\n'.join(out).rstrip() + '\n'

    def scan(self, node, subdir='.'):
        if False:
            while True:
                i = 10
        noopt_cre = re.compile('\\s*\\[.*$')
        if node.includes is not None:
            includes = node.includes
        else:
            text = self.canonical_text(node.get_text_contents())
            includes = self.cre.findall(text)
            split_includes = []
            for include in includes:
                inc_type = noopt_cre.sub('', include[0])
                inc_subdir = subdir
                if inc_type in self.two_arg_commands:
                    inc_subdir = os.path.join(subdir, include[1])
                    inc_list = include[2].split(',')
                else:
                    inc_list = include[1].split(',')
                for inc in inc_list:
                    split_includes.append((inc_type, inc_subdir, inc))
            includes = split_includes
            node.includes = includes
        return includes

    def scan_recurse(self, node, path=()):
        if False:
            for i in range(10):
                print('nop')
        ' do a recursive scan of the top level target file\n        This lets us search for included files based on the\n        directory of the main file just as latex does'
        path_dict = dict(list(path))
        queue = []
        queue.extend(self.scan(node))
        seen = {}
        nodes = []
        source_dir = node.get_dir()
        while queue:
            include = queue.pop()
            (inc_type, inc_subdir, inc_filename) = include
            try:
                if seen[inc_filename]:
                    continue
            except KeyError:
                seen[inc_filename] = True
            (n, i) = self.find_include(include, source_dir, path_dict)
            if n is None:
                if inc_type != 'usepackage':
                    SCons.Warnings.warn(SCons.Warnings.DependencyWarning, 'No dependency generated for file: %s (included from: %s) -- file not found' % (i, node))
            else:
                sortkey = self.sort_key(n)
                nodes.append((sortkey, n))
                queue.extend(self.scan(n, inc_subdir))
        return [pair[1] for pair in sorted(nodes)]