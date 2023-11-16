"""Base class of epub2/epub3 builders."""
from __future__ import annotations
import html
import os
import re
import time
from os import path
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import quote
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
from docutils import nodes
from docutils.utils import smartquotes
from sphinx import addnodes
from sphinx.builders.html import BuildInfo, StandaloneHTMLBuilder
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.display import status_iterator
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.osutil import copyfile, ensuredir, relpath
if TYPE_CHECKING:
    from docutils.nodes import Element, Node
try:
    from PIL import Image
except ImportError:
    Image = None
logger = logging.getLogger(__name__)
COVERPAGE_NAME = 'epub-cover.xhtml'
TOCTREE_TEMPLATE = 'toctree-l%d'
LINK_TARGET_TEMPLATE = ' [%(uri)s]'
FOOTNOTE_LABEL_TEMPLATE = '#%d'
FOOTNOTES_RUBRIC_NAME = 'Footnotes'
CSS_LINK_TARGET_CLASS = 'link-target'
GUIDE_TITLES = {'toc': 'Table of Contents', 'cover': 'Cover'}
MEDIA_TYPES = {'.xhtml': 'application/xhtml+xml', '.css': 'text/css', '.png': 'image/png', '.webp': 'image/webp', '.gif': 'image/gif', '.svg': 'image/svg+xml', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.otf': 'font/otf', '.ttf': 'font/ttf', '.woff': 'font/woff'}
VECTOR_GRAPHICS_EXTENSIONS = ('.svg',)
REFURI_RE = re.compile('([^#:]*#)(.*)')

class ManifestItem(NamedTuple):
    href: str
    id: str
    media_type: str

class Spine(NamedTuple):
    idref: str
    linear: bool

class Guide(NamedTuple):
    type: str
    title: str
    uri: str

class NavPoint(NamedTuple):
    navpoint: str
    playorder: int
    text: str
    refuri: str
    children: list[NavPoint]

def sphinx_smarty_pants(t: str, language: str='en') -> str:
    if False:
        while True:
            i = 10
    t = t.replace('&quot;', '"')
    t = smartquotes.educateDashesOldSchool(t)
    t = smartquotes.educateQuotes(t, language)
    t = t.replace('"', '&quot;')
    return t
ssp = sphinx_smarty_pants

class EpubBuilder(StandaloneHTMLBuilder):
    """
    Builder that outputs epub files.

    It creates the metainfo files container.opf, toc.ncx, mimetype, and
    META-INF/container.xml.  Afterwards, all necessary files are zipped to an
    epub file.
    """
    copysource = False
    supported_image_types = ['image/svg+xml', 'image/png', 'image/gif', 'image/jpeg']
    supported_remote_images = False
    add_permalinks = False
    allow_sharp_as_current_path = False
    embedded = True
    download_support = False
    html_scaled_image_link = False
    search = False
    coverpage_name = COVERPAGE_NAME
    toctree_template = TOCTREE_TEMPLATE
    link_target_template = LINK_TARGET_TEMPLATE
    css_link_target_class = CSS_LINK_TARGET_CLASS
    guide_titles = GUIDE_TITLES
    media_types = MEDIA_TYPES
    refuri_re = REFURI_RE
    template_dir = ''
    doctype = ''

    def init(self) -> None:
        if False:
            while True:
                i = 10
        super().init()
        self.out_suffix = '.xhtml'
        self.link_suffix = '.xhtml'
        self.playorder = 0
        self.tocid = 0
        self.id_cache: dict[str, str] = {}
        self.use_index = self.get_builder_config('use_index', 'epub')
        self.refnodes: list[dict[str, Any]] = []

    def create_build_info(self) -> BuildInfo:
        if False:
            i = 10
            return i + 15
        return BuildInfo(self.config, self.tags, ['html', 'epub'])

    def get_theme_config(self) -> tuple[str, dict]:
        if False:
            while True:
                i = 10
        return (self.config.epub_theme, self.config.epub_theme_options)

    def make_id(self, name: str) -> str:
        if False:
            print('Hello World!')
        'Return a unique id for name.'
        id = self.id_cache.get(name)
        if not id:
            id = 'epub-%d' % self.env.new_serialno('epub')
            self.id_cache[name] = id
        return id

    def get_refnodes(self, doctree: Node, result: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if False:
            print('Hello World!')
        'Collect section titles, their depth in the toc and the refuri.'
        if isinstance(doctree, nodes.reference) and doctree.get('refuri'):
            refuri = doctree['refuri']
            if refuri.startswith(('http://', 'https://', 'irc:', 'mailto:')):
                return result
            classes = doctree.parent.attributes['classes']
            for level in range(8, 0, -1):
                if self.toctree_template % level in classes:
                    result.append({'level': level, 'refuri': html.escape(refuri), 'text': ssp(html.escape(doctree.astext()))})
                    break
        elif isinstance(doctree, nodes.Element):
            for elem in doctree:
                result = self.get_refnodes(elem, result)
        return result

    def check_refnodes(self, nodes: list[dict[str, Any]]) -> None:
        if False:
            print('Hello World!')
        appeared: set[str] = set()
        for node in nodes:
            if node['refuri'] in appeared:
                logger.warning(__('duplicated ToC entry found: %s'), node['refuri'], type='epub', subtype='duplicated_toc_entry')
            else:
                appeared.add(node['refuri'])

    def get_toc(self) -> None:
        if False:
            while True:
                i = 10
        'Get the total table of contents, containing the root_doc\n        and pre and post files not managed by sphinx.\n        '
        doctree = self.env.get_and_resolve_doctree(self.config.root_doc, self, prune_toctrees=False, includehidden=True)
        self.refnodes = self.get_refnodes(doctree, [])
        master_dir = path.dirname(self.config.root_doc)
        if master_dir:
            master_dir += '/'
            for item in self.refnodes:
                item['refuri'] = master_dir + item['refuri']
        self.toc_add_files(self.refnodes)

    def toc_add_files(self, refnodes: list[dict[str, Any]]) -> None:
        if False:
            return 10
        'Add the root_doc, pre and post files to a list of refnodes.\n        '
        refnodes.insert(0, {'level': 1, 'refuri': html.escape(self.config.root_doc + self.out_suffix), 'text': ssp(html.escape(self.env.titles[self.config.root_doc].astext()))})
        for (file, text) in reversed(self.config.epub_pre_files):
            refnodes.insert(0, {'level': 1, 'refuri': html.escape(file), 'text': ssp(html.escape(text))})
        for (file, text) in self.config.epub_post_files:
            refnodes.append({'level': 1, 'refuri': html.escape(file), 'text': ssp(html.escape(text))})

    def fix_fragment(self, prefix: str, fragment: str) -> str:
        if False:
            print('Hello World!')
        'Return a href/id attribute with colons replaced by hyphens.'
        return prefix + fragment.replace(':', '-')

    def fix_ids(self, tree: nodes.document) -> None:
        if False:
            i = 10
            return i + 15
        'Replace colons with hyphens in href and id attributes.\n\n        Some readers crash because they interpret the part as a\n        transport protocol specification.\n        '

        def update_node_id(node: Element) -> None:
            if False:
                for i in range(10):
                    print('nop')
            'Update IDs of given *node*.'
            new_ids: list[str] = []
            for node_id in node['ids']:
                new_id = self.fix_fragment('', node_id)
                if new_id not in new_ids:
                    new_ids.append(new_id)
            node['ids'] = new_ids
        for reference in tree.findall(nodes.reference):
            if 'refuri' in reference:
                m = self.refuri_re.match(reference['refuri'])
                if m:
                    reference['refuri'] = self.fix_fragment(m.group(1), m.group(2))
            if 'refid' in reference:
                reference['refid'] = self.fix_fragment('', reference['refid'])
        for target in tree.findall(nodes.target):
            update_node_id(target)
            next_node: Node = target.next_node(ascend=True)
            if isinstance(next_node, nodes.Element):
                update_node_id(next_node)
        for desc_signature in tree.findall(addnodes.desc_signature):
            update_node_id(desc_signature)

    def add_visible_links(self, tree: nodes.document, show_urls: str='inline') -> None:
        if False:
            return 10
        'Add visible link targets for external links'

        def make_footnote_ref(doc: nodes.document, label: str) -> nodes.footnote_reference:
            if False:
                for i in range(10):
                    print('nop')
            'Create a footnote_reference node with children'
            footnote_ref = nodes.footnote_reference('[#]_')
            footnote_ref.append(nodes.Text(label))
            doc.note_autofootnote_ref(footnote_ref)
            return footnote_ref

        def make_footnote(doc: nodes.document, label: str, uri: str) -> nodes.footnote:
            if False:
                print('Hello World!')
            'Create a footnote node with children'
            footnote = nodes.footnote(uri)
            para = nodes.paragraph()
            para.append(nodes.Text(uri))
            footnote.append(para)
            footnote.insert(0, nodes.label('', label))
            doc.note_autofootnote(footnote)
            return footnote

        def footnote_spot(tree: nodes.document) -> tuple[Element, int]:
            if False:
                i = 10
                return i + 15
            'Find or create a spot to place footnotes.\n\n            The function returns the tuple (parent, index).'
            fns = list(tree.findall(nodes.footnote))
            if fns:
                fn = fns[-1]
                return (fn.parent, fn.parent.index(fn) + 1)
            for node in tree.findall(nodes.rubric):
                if len(node) == 1 and node.astext() == FOOTNOTES_RUBRIC_NAME:
                    return (node.parent, node.parent.index(node) + 1)
            doc = next(tree.findall(nodes.document))
            rub = nodes.rubric()
            rub.append(nodes.Text(FOOTNOTES_RUBRIC_NAME))
            doc.append(rub)
            return (doc, doc.index(rub) + 1)
        if show_urls == 'no':
            return
        if show_urls == 'footnote':
            doc = next(tree.findall(nodes.document))
            (fn_spot, fn_idx) = footnote_spot(tree)
            nr = 1
        for node in list(tree.findall(nodes.reference)):
            uri = node.get('refuri', '')
            if uri.startswith(('http:', 'https:', 'ftp:')) and uri not in node.astext():
                idx = node.parent.index(node) + 1
                if show_urls == 'inline':
                    uri = self.link_target_template % {'uri': uri}
                    link = nodes.inline(uri, uri)
                    link['classes'].append(self.css_link_target_class)
                    node.parent.insert(idx, link)
                elif show_urls == 'footnote':
                    label = FOOTNOTE_LABEL_TEMPLATE % nr
                    nr += 1
                    footnote_ref = make_footnote_ref(doc, label)
                    node.parent.insert(idx, footnote_ref)
                    footnote = make_footnote(doc, label, uri)
                    fn_spot.insert(fn_idx, footnote)
                    footnote_ref['refid'] = footnote['ids'][0]
                    footnote.add_backref(footnote_ref['ids'][0])
                    fn_idx += 1

    def write_doc(self, docname: str, doctree: nodes.document) -> None:
        if False:
            i = 10
            return i + 15
        'Write one document file.\n\n        This method is overwritten in order to fix fragment identifiers\n        and to add visible external links.\n        '
        self.fix_ids(doctree)
        self.add_visible_links(doctree, self.config.epub_show_urls)
        super().write_doc(docname, doctree)

    def fix_genindex(self, tree: list[tuple[str, list[tuple[str, Any]]]]) -> None:
        if False:
            return 10
        'Fix href attributes for genindex pages.'
        for (_key, columns) in tree:
            for (_entryname, (links, subitems, _key)) in columns:
                for (i, (ismain, link)) in enumerate(links):
                    m = self.refuri_re.match(link)
                    if m:
                        links[i] = (ismain, self.fix_fragment(m.group(1), m.group(2)))
                for (_subentryname, subentrylinks) in subitems:
                    for (i, (ismain, link)) in enumerate(subentrylinks):
                        m = self.refuri_re.match(link)
                        if m:
                            subentrylinks[i] = (ismain, self.fix_fragment(m.group(1), m.group(2)))

    def is_vector_graphics(self, filename: str) -> bool:
        if False:
            while True:
                i = 10
        'Does the filename extension indicate a vector graphic format?'
        ext = path.splitext(filename)[-1]
        return ext in VECTOR_GRAPHICS_EXTENSIONS

    def copy_image_files_pil(self) -> None:
        if False:
            i = 10
            return i + 15
        'Copy images using Pillow, the Python Imaging Library.\n        The method tries to read and write the files with Pillow, converting\n        the format and resizing the image if necessary/possible.\n        '
        ensuredir(path.join(self.outdir, self.imagedir))
        for src in status_iterator(self.images, __('copying images... '), 'brown', len(self.images), self.app.verbosity):
            dest = self.images[src]
            try:
                img = Image.open(path.join(self.srcdir, src))
            except OSError:
                if not self.is_vector_graphics(src):
                    logger.warning(__('cannot read image file %r: copying it instead'), path.join(self.srcdir, src))
                try:
                    copyfile(path.join(self.srcdir, src), path.join(self.outdir, self.imagedir, dest))
                except OSError as err:
                    logger.warning(__('cannot copy image file %r: %s'), path.join(self.srcdir, src), err)
                continue
            if self.config.epub_fix_images:
                if img.mode in ('P',):
                    img = img.convert()
            if self.config.epub_max_image_width > 0:
                (width, height) = img.size
                nw = self.config.epub_max_image_width
                if width > nw:
                    nh = round(height * nw / width)
                    img = img.resize((nw, nh), Image.BICUBIC)
            try:
                img.save(path.join(self.outdir, self.imagedir, dest))
            except OSError as err:
                logger.warning(__('cannot write image file %r: %s'), path.join(self.srcdir, src), err)

    def copy_image_files(self) -> None:
        if False:
            while True:
                i = 10
        'Copy image files to destination directory.\n        This overwritten method can use Pillow to convert image files.\n        '
        if self.images:
            if self.config.epub_fix_images or self.config.epub_max_image_width:
                if not Image:
                    logger.warning(__('Pillow not found - copying image files'))
                    super().copy_image_files()
                else:
                    self.copy_image_files_pil()
            else:
                super().copy_image_files()

    def copy_download_files(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def handle_page(self, pagename: str, addctx: dict, templatename: str='page.html', outfilename: str | None=None, event_arg: Any=None) -> None:
        if False:
            while True:
                i = 10
        'Create a rendered page.\n\n        This method is overwritten for genindex pages in order to fix href link\n        attributes.\n        '
        if pagename.startswith('genindex') and 'genindexentries' in addctx:
            if not self.use_index:
                return
            self.fix_genindex(addctx['genindexentries'])
        addctx['doctype'] = self.doctype
        super().handle_page(pagename, addctx, templatename, outfilename, event_arg)

    def build_mimetype(self) -> None:
        if False:
            return 10
        'Write the metainfo file mimetype.'
        logger.info(__('writing mimetype file...'))
        copy_asset_file(path.join(self.template_dir, 'mimetype'), self.outdir)

    def build_container(self, outname: str='META-INF/container.xml') -> None:
        if False:
            while True:
                i = 10
        'Write the metainfo file META-INF/container.xml.'
        logger.info(__('writing META-INF/container.xml file...'))
        outdir = path.join(self.outdir, 'META-INF')
        ensuredir(outdir)
        copy_asset_file(path.join(self.template_dir, 'container.xml'), outdir)

    def content_metadata(self) -> dict[str, Any]:
        if False:
            while True:
                i = 10
        'Create a dictionary with all metadata for the content.opf\n        file properly escaped.\n        '
        if (source_date_epoch := os.getenv('SOURCE_DATE_EPOCH')) is not None:
            time_tuple = time.gmtime(int(source_date_epoch))
        else:
            time_tuple = time.gmtime()
        metadata: dict[str, Any] = {}
        metadata['title'] = html.escape(self.config.epub_title)
        metadata['author'] = html.escape(self.config.epub_author)
        metadata['uid'] = html.escape(self.config.epub_uid)
        metadata['lang'] = html.escape(self.config.epub_language)
        metadata['publisher'] = html.escape(self.config.epub_publisher)
        metadata['copyright'] = html.escape(self.config.epub_copyright)
        metadata['scheme'] = html.escape(self.config.epub_scheme)
        metadata['id'] = html.escape(self.config.epub_identifier)
        metadata['date'] = html.escape(time.strftime('%Y-%m-%d', time_tuple))
        metadata['manifest_items'] = []
        metadata['spines'] = []
        metadata['guides'] = []
        return metadata

    def build_content(self) -> None:
        if False:
            i = 10
            return i + 15
        'Write the metainfo file content.opf It contains bibliographic data,\n        a file list and the spine (the reading order).\n        '
        logger.info(__('writing content.opf file...'))
        metadata = self.content_metadata()
        self.files: list[str] = []
        self.ignored_files = ['.buildinfo', 'mimetype', 'content.opf', 'toc.ncx', 'META-INF/container.xml', 'Thumbs.db', 'ehthumbs.db', '.DS_Store', 'nav.xhtml', self.config.epub_basename + '.epub'] + self.config.epub_exclude_files
        if not self.use_index:
            self.ignored_files.append('genindex' + self.out_suffix)
        for (root, dirs, files) in os.walk(self.outdir):
            dirs.sort()
            for fn in sorted(files):
                filename = relpath(path.join(root, fn), self.outdir)
                if filename in self.ignored_files:
                    continue
                ext = path.splitext(filename)[-1]
                if ext not in self.media_types:
                    if ext not in ('.js', '.xml'):
                        logger.warning(__('unknown mimetype for %s, ignoring'), filename, type='epub', subtype='unknown_project_files')
                    continue
                filename = filename.replace(os.sep, '/')
                item = ManifestItem(html.escape(quote(filename)), html.escape(self.make_id(filename)), html.escape(self.media_types[ext]))
                metadata['manifest_items'].append(item)
                self.files.append(filename)
        spinefiles = set()
        for refnode in self.refnodes:
            if '#' in refnode['refuri']:
                continue
            if refnode['refuri'] in self.ignored_files:
                continue
            spine = Spine(html.escape(self.make_id(refnode['refuri'])), True)
            metadata['spines'].append(spine)
            spinefiles.add(refnode['refuri'])
        for info in self.domain_indices:
            spine = Spine(html.escape(self.make_id(info[0] + self.out_suffix)), True)
            metadata['spines'].append(spine)
            spinefiles.add(info[0] + self.out_suffix)
        if self.use_index:
            spine = Spine(html.escape(self.make_id('genindex' + self.out_suffix)), True)
            metadata['spines'].append(spine)
            spinefiles.add('genindex' + self.out_suffix)
        for name in self.files:
            if name not in spinefiles and name.endswith(self.out_suffix):
                spine = Spine(html.escape(self.make_id(name)), False)
                metadata['spines'].append(spine)
        html_tmpl = None
        if self.config.epub_cover:
            (image, html_tmpl) = self.config.epub_cover
            image = image.replace(os.sep, '/')
            metadata['cover'] = html.escape(self.make_id(image))
            if html_tmpl:
                spine = Spine(html.escape(self.make_id(self.coverpage_name)), True)
                metadata['spines'].insert(0, spine)
                if self.coverpage_name not in self.files:
                    ext = path.splitext(self.coverpage_name)[-1]
                    self.files.append(self.coverpage_name)
                    item = ManifestItem(html.escape(self.coverpage_name), html.escape(self.make_id(self.coverpage_name)), html.escape(self.media_types[ext]))
                    metadata['manifest_items'].append(item)
                ctx = {'image': html.escape(image), 'title': self.config.project}
                self.handle_page(path.splitext(self.coverpage_name)[0], ctx, html_tmpl)
                spinefiles.add(self.coverpage_name)
        auto_add_cover = True
        auto_add_toc = True
        if self.config.epub_guide:
            for (type, uri, title) in self.config.epub_guide:
                file = uri.split('#')[0]
                if file not in self.files:
                    self.files.append(file)
                if type == 'cover':
                    auto_add_cover = False
                if type == 'toc':
                    auto_add_toc = False
                metadata['guides'].append(Guide(html.escape(type), html.escape(title), html.escape(uri)))
        if auto_add_cover and html_tmpl:
            metadata['guides'].append(Guide('cover', self.guide_titles['cover'], html.escape(self.coverpage_name)))
        if auto_add_toc and self.refnodes:
            metadata['guides'].append(Guide('toc', self.guide_titles['toc'], html.escape(self.refnodes[0]['refuri'])))
        copy_asset_file(path.join(self.template_dir, 'content.opf_t'), self.outdir, metadata)

    def new_navpoint(self, node: dict[str, Any], level: int, incr: bool=True) -> NavPoint:
        if False:
            while True:
                i = 10
        'Create a new entry in the toc from the node at given level.'
        if incr:
            self.playorder += 1
        self.tocid += 1
        return NavPoint('navPoint%d' % self.tocid, self.playorder, node['text'], node['refuri'], [])

    def build_navpoints(self, nodes: list[dict[str, Any]]) -> list[NavPoint]:
        if False:
            i = 10
            return i + 15
        'Create the toc navigation structure.\n\n        Subelements of a node are nested inside the navpoint.  For nested nodes\n        the parent node is reinserted in the subnav.\n        '
        navstack: list[NavPoint] = []
        navstack.append(NavPoint('dummy', 0, '', '', []))
        level = 0
        lastnode = None
        for node in nodes:
            if not node['text']:
                continue
            file = node['refuri'].split('#')[0]
            if file in self.ignored_files:
                continue
            if node['level'] > self.config.epub_tocdepth:
                continue
            if node['level'] == level:
                navpoint = self.new_navpoint(node, level)
                navstack.pop()
                navstack[-1].children.append(navpoint)
                navstack.append(navpoint)
            elif node['level'] == level + 1:
                level += 1
                if lastnode and self.config.epub_tocdup:
                    navstack[-1].children.append(self.new_navpoint(lastnode, level, False))
                navpoint = self.new_navpoint(node, level)
                navstack[-1].children.append(navpoint)
                navstack.append(navpoint)
            elif node['level'] < level:
                while node['level'] < len(navstack):
                    navstack.pop()
                level = node['level']
                navpoint = self.new_navpoint(node, level)
                navstack[-1].children.append(navpoint)
                navstack.append(navpoint)
            else:
                raise
            lastnode = node
        return navstack[0].children

    def toc_metadata(self, level: int, navpoints: list[NavPoint]) -> dict[str, Any]:
        if False:
            while True:
                i = 10
        'Create a dictionary with all metadata for the toc.ncx file\n        properly escaped.\n        '
        metadata: dict[str, Any] = {}
        metadata['uid'] = self.config.epub_uid
        metadata['title'] = html.escape(self.config.epub_title)
        metadata['level'] = level
        metadata['navpoints'] = navpoints
        return metadata

    def build_toc(self) -> None:
        if False:
            while True:
                i = 10
        'Write the metainfo file toc.ncx.'
        logger.info(__('writing toc.ncx file...'))
        if self.config.epub_tocscope == 'default':
            doctree = self.env.get_and_resolve_doctree(self.config.root_doc, self, prune_toctrees=False, includehidden=False)
            refnodes = self.get_refnodes(doctree, [])
            self.toc_add_files(refnodes)
        else:
            refnodes = self.refnodes
        self.check_refnodes(refnodes)
        navpoints = self.build_navpoints(refnodes)
        level = max((item['level'] for item in self.refnodes))
        level = min(level, self.config.epub_tocdepth)
        copy_asset_file(path.join(self.template_dir, 'toc.ncx_t'), self.outdir, self.toc_metadata(level, navpoints))

    def build_epub(self) -> None:
        if False:
            return 10
        'Write the epub file.\n\n        It is a zip file with the mimetype file stored uncompressed as the first\n        entry.\n        '
        outname = self.config.epub_basename + '.epub'
        logger.info(__('writing %s file...'), outname)
        epub_filename = path.join(self.outdir, outname)
        with ZipFile(epub_filename, 'w', ZIP_DEFLATED) as epub:
            epub.write(path.join(self.outdir, 'mimetype'), 'mimetype', ZIP_STORED)
            for filename in ('META-INF/container.xml', 'content.opf', 'toc.ncx'):
                epub.write(path.join(self.outdir, filename), filename, ZIP_DEFLATED)
            for filename in self.files:
                epub.write(path.join(self.outdir, filename), filename, ZIP_DEFLATED)