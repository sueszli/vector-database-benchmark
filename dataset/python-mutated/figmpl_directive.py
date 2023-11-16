"""
Add a ``figure-mpl`` directive that is a responsive version of ``figure``.

This implementation is very similar to ``.. figure::``, except it also allows a
``srcset=`` argument to be passed to the image tag, hence allowing responsive
resolution images.

There is no particular reason this could not be used standalone, but is meant
to be used with :doc:`/api/sphinxext_plot_directive_api`.

Note that the directory organization is a bit different than ``.. figure::``.
See the *FigureMpl* documentation below.

"""
from docutils import nodes
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.images import Figure, Image
import os
from os.path import relpath
from pathlib import PurePath, Path
import shutil
from sphinx.errors import ExtensionError
import matplotlib

class figmplnode(nodes.General, nodes.Element):
    pass

class FigureMpl(Figure):
    """
    Implements a directive to allow an optional hidpi image.

    Meant to be used with the *plot_srcset* configuration option in conf.py,
    and gets set in the TEMPLATE of plot_directive.py

    e.g.::

        .. figure-mpl:: plot_directive/some_plots-1.png
            :alt: bar
            :srcset: plot_directive/some_plots-1.png,
                     plot_directive/some_plots-1.2x.png 2.00x
            :class: plot-directive

    The resulting html (at ``some_plots.html``) is::

        <img src="sphx_glr_bar_001_hidpi.png"
            srcset="_images/some_plot-1.png,
                    _images/some_plots-1.2x.png 2.00x",
            alt="bar"
            class="plot_directive" />

    Note that the handling of subdirectories is different than that used by the sphinx
    figure directive::

        .. figure-mpl:: plot_directive/nestedpage/index-1.png
            :alt: bar
            :srcset: plot_directive/nestedpage/index-1.png
                     plot_directive/nestedpage/index-1.2x.png 2.00x
            :class: plot_directive

    The resulting html (at ``nestedpage/index.html``)::

        <img src="../_images/nestedpage-index-1.png"
            srcset="../_images/nestedpage-index-1.png,
                    ../_images/_images/nestedpage-index-1.2x.png 2.00x",
            alt="bar"
            class="sphx-glr-single-img" />

    where the subdirectory is included in the image name for uniqueness.
    """
    has_content = False
    required_arguments = 1
    optional_arguments = 2
    final_argument_whitespace = False
    option_spec = {'alt': directives.unchanged, 'height': directives.length_or_unitless, 'width': directives.length_or_percentage_or_unitless, 'scale': directives.nonnegative_int, 'align': Image.align, 'class': directives.class_option, 'caption': directives.unchanged, 'srcset': directives.unchanged}

    def run(self):
        if False:
            i = 10
            return i + 15
        image_node = figmplnode()
        imagenm = self.arguments[0]
        image_node['alt'] = self.options.get('alt', '')
        image_node['align'] = self.options.get('align', None)
        image_node['class'] = self.options.get('class', None)
        image_node['width'] = self.options.get('width', None)
        image_node['height'] = self.options.get('height', None)
        image_node['scale'] = self.options.get('scale', None)
        image_node['caption'] = self.options.get('caption', None)
        image_node['uri'] = imagenm
        image_node['srcset'] = self.options.get('srcset', None)
        return [image_node]

def _parse_srcsetNodes(st):
    if False:
        print('Hello World!')
    '\n    parse srcset...\n    '
    entries = st.split(',')
    srcset = {}
    for entry in entries:
        spl = entry.strip().split(' ')
        if len(spl) == 1:
            srcset[0] = spl[0]
        elif len(spl) == 2:
            mult = spl[1][:-1]
            srcset[float(mult)] = spl[0]
        else:
            raise ExtensionError(f'srcset argument "{entry}" is invalid.')
    return srcset

def _copy_images_figmpl(self, node):
    if False:
        return 10
    if node['srcset']:
        srcset = _parse_srcsetNodes(node['srcset'])
    else:
        srcset = None
    docsource = PurePath(self.document['source']).parent
    srctop = self.builder.srcdir
    rel = relpath(docsource, srctop).replace('.', '').replace(os.sep, '-')
    if len(rel):
        rel += '-'
    imagedir = PurePath(self.builder.outdir, self.builder.imagedir)
    Path(imagedir).mkdir(parents=True, exist_ok=True)
    if srcset:
        for src in srcset.values():
            abspath = PurePath(docsource, src)
            name = rel + abspath.name
            shutil.copyfile(abspath, imagedir / name)
    else:
        abspath = PurePath(docsource, node['uri'])
        name = rel + abspath.name
        shutil.copyfile(abspath, imagedir / name)
    return (imagedir, srcset, rel)

def visit_figmpl_html(self, node):
    if False:
        for i in range(10):
            print('nop')
    (imagedir, srcset, rel) = _copy_images_figmpl(self, node)
    docsource = PurePath(self.document['source'])
    srctop = PurePath(self.builder.srcdir, '')
    relsource = relpath(docsource, srctop)
    desttop = PurePath(self.builder.outdir, '')
    dest = desttop / relsource
    imagerel = PurePath(relpath(imagedir, dest.parent)).as_posix()
    if self.builder.name == 'dirhtml':
        imagerel = f'..{imagerel}'
    nm = PurePath(node['uri'][1:]).name
    uri = f'{imagerel}/{rel}{nm}'
    maxsrc = uri
    srcsetst = ''
    if srcset:
        maxmult = -1
        for (mult, src) in srcset.items():
            nm = PurePath(src[1:]).name
            path = f'{imagerel}/{rel}{nm}'
            srcsetst += path
            if mult == 0:
                srcsetst += ', '
            else:
                srcsetst += f' {mult:1.2f}x, '
            if mult > maxmult:
                maxmult = mult
                maxsrc = path
        srcsetst = srcsetst[:-2]
    alt = node['alt']
    if node['class'] is not None:
        classst = ' '.join(node['class'])
        classst = f'class="{classst}"'
    else:
        classst = ''
    stylers = ['width', 'height', 'scale']
    stylest = ''
    for style in stylers:
        if node[style]:
            stylest += f'{style}: {node[style]};'
    figalign = node['align'] if node['align'] else 'center'
    img_block = f'<img src="{uri}" style="{stylest}" srcset="{srcsetst}" alt="{alt}" {classst}/>'
    html_block = f'<figure class="align-{figalign}">\n'
    html_block += f'  <a class="reference internal image-reference" href="{maxsrc}">\n'
    html_block += f'    {img_block}\n  </a>\n'
    if node['caption']:
        html_block += '  <figcaption>\n'
        html_block += f"""   <p><span class="caption-text">{node['caption']}</span></p>\n"""
        html_block += '  </figcaption>\n'
    html_block += '</figure>\n'
    self.body.append(html_block)

def visit_figmpl_latex(self, node):
    if False:
        while True:
            i = 10
    if node['srcset'] is not None:
        (imagedir, srcset) = _copy_images_figmpl(self, node)
        maxmult = -1
        maxmult = max(srcset, default=-1)
        node['uri'] = PurePath(srcset[maxmult]).name
    self.visit_figure(node)

def depart_figmpl_html(self, node):
    if False:
        for i in range(10):
            print('nop')
    pass

def depart_figmpl_latex(self, node):
    if False:
        print('Hello World!')
    self.depart_figure(node)

def figurempl_addnode(app):
    if False:
        while True:
            i = 10
    app.add_node(figmplnode, html=(visit_figmpl_html, depart_figmpl_html), latex=(visit_figmpl_latex, depart_figmpl_latex))

def setup(app):
    if False:
        print('Hello World!')
    app.add_directive('figure-mpl', FigureMpl)
    figurempl_addnode(app)
    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True, 'version': matplotlib.__version__}
    return metadata