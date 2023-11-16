"""
Sphinx plugin to run example scripts and create a gallery page.

Lightly modified from the mpld3 project.

"""
import os
import os.path as op
import re
import glob
import token
import tokenize
import shutil
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def execfile(filename, globals=None, locals=None):
    if False:
        return 10
    with open(filename, 'rb') as fp:
        exec(compile(fp.read(), filename, 'exec'), globals, locals)
RST_TEMPLATE = '\n\n.. currentmodule:: seaborn\n\n.. _{sphinx_tag}:\n\n{docstring}\n\n.. image:: {img_file}\n\n**seaborn components used:** {components}\n\n.. literalinclude:: {fname}\n    :lines: {end_line}-\n\n'
INDEX_TEMPLATE = '\n:html_theme.sidebar_secondary.remove:\n\n.. raw:: html\n\n    <style type="text/css">\n    .thumb {{\n        position: relative;\n        float: left;\n        width: 180px;\n        height: 180px;\n        margin: 0;\n    }}\n\n    .thumb img {{\n        position: absolute;\n        display: inline;\n        left: 0;\n        width: 170px;\n        height: 170px;\n        opacity:1.0;\n        filter:alpha(opacity=100); /* For IE8 and earlier */\n    }}\n\n    .thumb:hover img {{\n        -webkit-filter: blur(3px);\n        -moz-filter: blur(3px);\n        -o-filter: blur(3px);\n        -ms-filter: blur(3px);\n        filter: blur(3px);\n        opacity:1.0;\n        filter:alpha(opacity=100); /* For IE8 and earlier */\n    }}\n\n    .thumb span {{\n        position: absolute;\n        display: inline;\n        left: 0;\n        width: 170px;\n        height: 170px;\n        background: #000;\n        color: #fff;\n        visibility: hidden;\n        opacity: 0;\n        z-index: 100;\n    }}\n\n    .thumb p {{\n        position: absolute;\n        top: 45%;\n        width: 170px;\n        font-size: 110%;\n        color: #fff;\n    }}\n\n    .thumb:hover span {{\n        visibility: visible;\n        opacity: .4;\n    }}\n\n    .caption {{\n        position: absolute;\n        width: 180px;\n        top: 170px;\n        text-align: center !important;\n    }}\n    </style>\n\n.. _{sphinx_tag}:\n\nExample gallery\n===============\n\n{toctree}\n\n{contents}\n\n.. raw:: html\n\n    <div style="clear: both"></div>\n'

def create_thumbnail(infile, thumbfile, width=275, height=275, cx=0.5, cy=0.5, border=4):
    if False:
        for i in range(10):
            print('nop')
    (baseout, extout) = op.splitext(thumbfile)
    im = matplotlib.image.imread(infile)
    (rows, cols) = im.shape[:2]
    x0 = int(cx * cols - 0.5 * width)
    y0 = int(cy * rows - 0.5 * height)
    xslice = slice(x0, x0 + width)
    yslice = slice(y0, y0 + height)
    thumb = im[yslice, xslice]
    thumb[:border, :, :3] = thumb[-border:, :, :3] = 0
    thumb[:, :border, :3] = thumb[:, -border:, :3] = 0
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], aspect='auto', frameon=False, xticks=[], yticks=[])
    if all(thumb.shape):
        ax.imshow(thumb, aspect='auto', resample=True, interpolation='bilinear')
    else:
        warnings.warn(f'Bad thumbnail crop. {thumbfile} will be empty.')
    fig.savefig(thumbfile, dpi=dpi)
    return fig

def indent(s, N=4):
    if False:
        i = 10
        return i + 15
    'indent a string'
    return s.replace('\n', '\n' + N * ' ')

class ExampleGenerator:
    """Tools for generating an example page from a file"""

    def __init__(self, filename, target_dir):
        if False:
            return 10
        self.filename = filename
        self.target_dir = target_dir
        self.thumbloc = (0.5, 0.5)
        self.extract_docstring()
        with open(filename) as fid:
            self.filetext = fid.read()
        outfilename = op.join(target_dir, self.rstfilename)
        file_mtime = op.getmtime(filename)
        if not op.exists(outfilename) or op.getmtime(outfilename) < file_mtime:
            self.exec_file()
        else:
            print(f'skipping {self.filename}')

    @property
    def dirname(self):
        if False:
            return 10
        return op.split(self.filename)[0]

    @property
    def fname(self):
        if False:
            i = 10
            return i + 15
        return op.split(self.filename)[1]

    @property
    def modulename(self):
        if False:
            while True:
                i = 10
        return op.splitext(self.fname)[0]

    @property
    def pyfilename(self):
        if False:
            for i in range(10):
                print('nop')
        return self.modulename + '.py'

    @property
    def rstfilename(self):
        if False:
            return 10
        return self.modulename + '.rst'

    @property
    def htmlfilename(self):
        if False:
            print('Hello World!')
        return self.modulename + '.html'

    @property
    def pngfilename(self):
        if False:
            return 10
        pngfile = self.modulename + '.png'
        return '_images/' + pngfile

    @property
    def thumbfilename(self):
        if False:
            return 10
        pngfile = self.modulename + '_thumb.png'
        return pngfile

    @property
    def sphinxtag(self):
        if False:
            print('Hello World!')
        return self.modulename

    @property
    def pagetitle(self):
        if False:
            while True:
                i = 10
        return self.docstring.strip().split('\n')[0].strip()

    @property
    def plotfunc(self):
        if False:
            return 10
        match = re.search('sns\\.(.+plot)\\(', self.filetext)
        if match:
            return match.group(1)
        match = re.search('sns\\.(.+map)\\(', self.filetext)
        if match:
            return match.group(1)
        match = re.search('sns\\.(.+Grid)\\(', self.filetext)
        if match:
            return match.group(1)
        return ''

    @property
    def components(self):
        if False:
            print('Hello World!')
        objects = re.findall('sns\\.(\\w+)\\(', self.filetext)
        refs = []
        for obj in objects:
            if obj[0].isupper():
                refs.append(f':class:`{obj}`')
            else:
                refs.append(f':func:`{obj}`')
        return ', '.join(refs)

    def extract_docstring(self):
        if False:
            i = 10
            return i + 15
        ' Extract a module-level docstring\n        '
        lines = open(self.filename).readlines()
        start_row = 0
        if lines[0].startswith('#!'):
            lines.pop(0)
            start_row = 1
        docstring = ''
        first_par = ''
        line_iter = lines.__iter__()
        tokens = tokenize.generate_tokens(lambda : next(line_iter))
        for (tok_type, tok_content, _, (erow, _), _) in tokens:
            tok_type = token.tok_name[tok_type]
            if tok_type in ('NEWLINE', 'COMMENT', 'NL', 'INDENT', 'DEDENT'):
                continue
            elif tok_type == 'STRING':
                docstring = eval(tok_content)
                paragraphs = '\n'.join((line.rstrip() for line in docstring.split('\n'))).split('\n\n')
                if len(paragraphs) > 0:
                    first_par = paragraphs[0]
            break
        thumbloc = None
        for (i, line) in enumerate(docstring.split('\n')):
            m = re.match('^_thumb: (\\.\\d+),\\s*(\\.\\d+)', line)
            if m:
                thumbloc = (float(m.group(1)), float(m.group(2)))
                break
        if thumbloc is not None:
            self.thumbloc = thumbloc
            docstring = '\n'.join([l for l in docstring.split('\n') if not l.startswith('_thumb')])
        self.docstring = docstring
        self.short_desc = first_par
        self.end_line = erow + 1 + start_row

    def exec_file(self):
        if False:
            return 10
        print(f'running {self.filename}')
        plt.close('all')
        my_globals = {'pl': plt, 'plt': plt}
        execfile(self.filename, my_globals)
        fig = plt.gcf()
        fig.canvas.draw()
        pngfile = op.join(self.target_dir, self.pngfilename)
        thumbfile = op.join('example_thumbs', self.thumbfilename)
        self.html = f'<img src=../{self.pngfilename}>'
        fig.savefig(pngfile, dpi=75, bbox_inches='tight')
        (cx, cy) = self.thumbloc
        create_thumbnail(pngfile, thumbfile, cx=cx, cy=cy)

    def toctree_entry(self):
        if False:
            while True:
                i = 10
        return f'   ./{op.splitext(self.htmlfilename)[0]}\n\n'

    def contents_entry(self):
        if False:
            i = 10
            return i + 15
        return ".. raw:: html\n\n    <div class='thumb align-center'>\n    <a href=./{}>\n    <img src=../_static/{}>\n    <span class='thumb-label'>\n    <p>{}</p>\n    </span>\n    </a>\n    </div>\n\n\n\n".format(self.htmlfilename, self.thumbfilename, self.plotfunc)

def main(app):
    if False:
        for i in range(10):
            print('nop')
    static_dir = op.join(app.builder.srcdir, '_static')
    target_dir = op.join(app.builder.srcdir, 'examples')
    image_dir = op.join(app.builder.srcdir, 'examples/_images')
    thumb_dir = op.join(app.builder.srcdir, 'example_thumbs')
    source_dir = op.abspath(op.join(app.builder.srcdir, '..', 'examples'))
    if not op.exists(static_dir):
        os.makedirs(static_dir)
    if not op.exists(target_dir):
        os.makedirs(target_dir)
    if not op.exists(image_dir):
        os.makedirs(image_dir)
    if not op.exists(thumb_dir):
        os.makedirs(thumb_dir)
    if not op.exists(source_dir):
        os.makedirs(source_dir)
    banner_data = []
    toctree = '\n\n.. toctree::\n   :hidden:\n\n'
    contents = '\n\n'
    for filename in sorted(glob.glob(op.join(source_dir, '*.py'))):
        ex = ExampleGenerator(filename, target_dir)
        banner_data.append({'title': ex.pagetitle, 'url': op.join('examples', ex.htmlfilename), 'thumb': op.join(ex.thumbfilename)})
        shutil.copyfile(filename, op.join(target_dir, ex.pyfilename))
        output = RST_TEMPLATE.format(sphinx_tag=ex.sphinxtag, docstring=ex.docstring, end_line=ex.end_line, components=ex.components, fname=ex.pyfilename, img_file=ex.pngfilename)
        with open(op.join(target_dir, ex.rstfilename), 'w') as f:
            f.write(output)
        toctree += ex.toctree_entry()
        contents += ex.contents_entry()
    if len(banner_data) < 10:
        banner_data = (4 * banner_data)[:10]
    index_file = op.join(target_dir, 'index.rst')
    with open(index_file, 'w') as index:
        index.write(INDEX_TEMPLATE.format(sphinx_tag='example_gallery', toctree=toctree, contents=contents))

def setup(app):
    if False:
        for i in range(10):
            print('nop')
    app.connect('builder-inited', main)