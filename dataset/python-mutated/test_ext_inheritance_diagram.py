"""Test sphinx.ext.inheritance_diagram extension."""
import os
import re
import sys
import zlib
import pytest
from sphinx.ext.inheritance_diagram import InheritanceDiagram, InheritanceException, import_classes
from sphinx.ext.intersphinx import load_mappings, normalize_intersphinx_mapping

@pytest.mark.sphinx(buildername='html', testroot='inheritance')
@pytest.mark.usefixtures('if_graphviz_found')
def test_inheritance_diagram(app, status, warning):
    if False:
        for i in range(10):
            print('nop')
    orig_run = InheritanceDiagram.run
    graphs = {}

    def new_run(self):
        if False:
            return 10
        result = orig_run(self)
        node = result[0]
        source = os.path.basename(node.document.current_source).replace('.rst', '')
        graphs[source] = node['graph']
        return result
    InheritanceDiagram.run = new_run
    try:
        app.builder.build_all()
    finally:
        InheritanceDiagram.run = orig_run
    assert app.statuscode == 0
    html_warnings = warning.getvalue()
    assert html_warnings == ''
    for cls in graphs['basic_diagram'].class_info:
        assert cls in [('dummy.test.A', 'dummy.test.A', [], None), ('dummy.test.F', 'dummy.test.F', ['dummy.test.C'], None), ('dummy.test.C', 'dummy.test.C', ['dummy.test.A'], None), ('dummy.test.E', 'dummy.test.E', ['dummy.test.B'], None), ('dummy.test.D', 'dummy.test.D', ['dummy.test.B', 'dummy.test.C'], None), ('dummy.test.B', 'dummy.test.B', ['dummy.test.A'], None)]
    for cls in graphs['diagram_w_parts'].class_info:
        assert cls in [('A', 'dummy.test.A', [], None), ('F', 'dummy.test.F', ['C'], None), ('C', 'dummy.test.C', ['A'], None), ('E', 'dummy.test.E', ['B'], None), ('D', 'dummy.test.D', ['B', 'C'], None), ('B', 'dummy.test.B', ['A'], None)]
    for cls in graphs['diagram_w_1_top_class'].class_info:
        assert cls in [('dummy.test.A', 'dummy.test.A', [], None), ('dummy.test.F', 'dummy.test.F', ['dummy.test.C'], None), ('dummy.test.C', 'dummy.test.C', ['dummy.test.A'], None), ('dummy.test.E', 'dummy.test.E', ['dummy.test.B'], None), ('dummy.test.D', 'dummy.test.D', ['dummy.test.B', 'dummy.test.C'], None), ('dummy.test.B', 'dummy.test.B', [], None)]
    for cls in graphs['diagram_w_2_top_classes'].class_info:
        assert cls in [('dummy.test.F', 'dummy.test.F', ['dummy.test.C'], None), ('dummy.test.C', 'dummy.test.C', [], None), ('dummy.test.E', 'dummy.test.E', ['dummy.test.B'], None), ('dummy.test.D', 'dummy.test.D', ['dummy.test.B', 'dummy.test.C'], None), ('dummy.test.B', 'dummy.test.B', [], None)]
    for cls in graphs['diagram_module_w_2_top_classes'].class_info:
        assert cls in [('dummy.test.F', 'dummy.test.F', ['dummy.test.C'], None), ('dummy.test.C', 'dummy.test.C', [], None), ('dummy.test.E', 'dummy.test.E', ['dummy.test.B'], None), ('dummy.test.D', 'dummy.test.D', ['dummy.test.B', 'dummy.test.C'], None), ('dummy.test.B', 'dummy.test.B', [], None), ('dummy.test.A', 'dummy.test.A', [], None)]
    for cls in graphs['diagram_w_nested_classes'].class_info:
        assert cls in [('dummy.test_nested.A', 'dummy.test_nested.A', [], None), ('dummy.test_nested.C', 'dummy.test_nested.C', ['dummy.test_nested.A.B'], None), ('dummy.test_nested.A.B', 'dummy.test_nested.A.B', [], None)]
external_inventory = b'# Sphinx inventory version 2\n# Project: external\n# Version: 1.0\n# The remainder of this file is compressed using zlib.\n' + zlib.compress(b'external.other.Bob py:class 1 foo.html#external.other.Bob -\n')

@pytest.mark.sphinx('html', testroot='ext-inheritance_diagram')
@pytest.mark.usefixtures('if_graphviz_found')
def test_inheritance_diagram_png_html(tmp_path, app):
    if False:
        i = 10
        return i + 15
    inv_file = tmp_path / 'inventory'
    inv_file.write_bytes(external_inventory)
    app.config.intersphinx_mapping = {'https://example.org': str(inv_file)}
    app.config.intersphinx_cache_limit = 0
    normalize_intersphinx_mapping(app, app.config)
    load_mappings(app)
    app.builder.build_all()
    content = (app.outdir / 'index.html').read_text(encoding='utf8')
    base_maps = re.findall('<map .+\n.+\n</map>', content)
    pattern = '<figure class="align-default" id="id1">\n<div class="graphviz"><img src="_images/inheritance-\\w+.png" alt="Inheritance diagram of test.Foo" class="inheritance graphviz" /></div>\n<figcaption>\n<p><span class="caption-text">Test Foo!</span><a class="headerlink" href="#id1" title="Link to this image">¶</a></p>\n</figcaption>\n</figure>\n'
    assert re.search(pattern, content, re.M)
    subdir_content = (app.outdir / 'subdir/page1.html').read_text(encoding='utf8')
    subdir_maps = re.findall('<map .+\n.+\n</map>', subdir_content)
    subdir_maps = [re.sub('href="(\\S+)"', 'href="subdir/\\g<1>"', s) for s in subdir_maps]
    for diagram_content in base_maps + subdir_maps:
        if 'subdir.' in diagram_content:
            assert 'https://example.org' in diagram_content
        for href in re.findall('href="(\\S+?)"', diagram_content):
            if '://' in href:
                assert href.startswith('https://example.org/')
            else:
                reluri = href.rsplit('#', 1)[0]
                assert (app.outdir / reluri).exists()

@pytest.mark.sphinx('html', testroot='ext-inheritance_diagram', confoverrides={'graphviz_output_format': 'svg'})
@pytest.mark.usefixtures('if_graphviz_found')
def test_inheritance_diagram_svg_html(tmp_path, app):
    if False:
        for i in range(10):
            print('nop')
    inv_file = tmp_path / 'inventory'
    inv_file.write_bytes(external_inventory)
    app.config.intersphinx_mapping = {'subdir': ('https://example.org', str(inv_file))}
    app.config.intersphinx_cache_limit = 0
    normalize_intersphinx_mapping(app, app.config)
    load_mappings(app)
    app.builder.build_all()
    content = (app.outdir / 'index.html').read_text(encoding='utf8')
    base_svgs = re.findall('<object data="(_images/inheritance-\\w+.svg?)"', content)
    pattern = '<figure class="align-default" id="id1">\n<div class="graphviz"><object data="_images/inheritance-\\w+.svg" type="image/svg\\+xml" class="inheritance graphviz">\n<p class="warning">Inheritance diagram of test.Foo</p></object></div>\n<figcaption>\n<p><span class="caption-text">Test Foo!</span><a class="headerlink" href="#id1" title="Link to this image">¶</a></p>\n</figcaption>\n</figure>\n'
    assert re.search(pattern, content, re.M)
    subdir_content = (app.outdir / 'subdir/page1.html').read_text(encoding='utf8')
    subdir_svgs = re.findall('<object data="../(_images/inheritance-\\w+.svg?)"', subdir_content)
    for diagram in base_svgs + subdir_svgs:
        diagram_content = (app.outdir / diagram).read_text(encoding='utf8')
        if 'subdir.' in diagram_content:
            assert 'https://example.org' in diagram_content
        for href in re.findall('href="(\\S+?)"', diagram_content):
            if '://' in href:
                assert href.startswith('https://example.org/')
            else:
                reluri = href.rsplit('#', 1)[0]
                abs_uri = (app.outdir / app.builder.imagedir / reluri).resolve()
                assert abs_uri.exists()

@pytest.mark.sphinx('latex', testroot='ext-inheritance_diagram')
@pytest.mark.usefixtures('if_graphviz_found')
def test_inheritance_diagram_latex(app, status, warning):
    if False:
        return 10
    app.builder.build_all()
    content = (app.outdir / 'python.tex').read_text(encoding='utf8')
    pattern = '\\\\begin{figure}\\[htbp]\n\\\\centering\n\\\\capstart\n\n\\\\sphinxincludegraphics\\[\\]{inheritance-\\w+.pdf}\n\\\\caption{Test Foo!}\\\\label{\\\\detokenize{index:id1}}\\\\end{figure}'
    assert re.search(pattern, content, re.M)

@pytest.mark.sphinx('html', testroot='ext-inheritance_diagram', srcdir='ext-inheritance_diagram-alias')
@pytest.mark.usefixtures('if_graphviz_found')
def test_inheritance_diagram_latex_alias(app, status, warning):
    if False:
        print('Hello World!')
    app.config.inheritance_alias = {'test.Foo': 'alias.Foo'}
    app.builder.build_all()
    doc = app.env.get_and_resolve_doctree('index', app)
    aliased_graph = doc.children[0].children[3]['graph'].class_info
    assert len(aliased_graph) == 4
    assert ('test.DocSubDir2', 'test.DocSubDir2', ['test.DocSubDir1'], None) in aliased_graph
    assert ('test.DocSubDir1', 'test.DocSubDir1', ['test.DocHere'], None) in aliased_graph
    assert ('test.DocHere', 'test.DocHere', ['alias.Foo'], None) in aliased_graph
    assert ('alias.Foo', 'alias.Foo', [], None) in aliased_graph
    content = (app.outdir / 'index.html').read_text(encoding='utf8')
    pattern = '<figure class="align-default" id="id1">\n<div class="graphviz"><img src="_images/inheritance-\\w+.png" alt="Inheritance diagram of test.Foo" class="inheritance graphviz" /></div>\n<figcaption>\n<p><span class="caption-text">Test Foo!</span><a class="headerlink" href="#id1" title="Link to this image">¶</a></p>\n</figcaption>\n</figure>\n'
    assert re.search(pattern, content, re.M)

def test_import_classes(rootdir):
    if False:
        print('Hello World!')
    from sphinx.parsers import Parser, RSTParser
    from sphinx.util.i18n import CatalogInfo
    try:
        sys.path.append(str(rootdir / 'test-ext-inheritance_diagram'))
        from example.sphinx import DummyClass
        with pytest.raises(InheritanceException):
            import_classes('unknown', None)
        with pytest.raises(InheritanceException):
            import_classes('unknown.Unknown', None)
        with pytest.raises(InheritanceException):
            import_classes('unknown', '.')
        with pytest.raises(InheritanceException):
            import_classes('unknown.Unknown', '.')
        with pytest.raises(InheritanceException):
            import_classes('.', None)
        classes = import_classes('sphinx', None)
        assert classes == []
        classes = import_classes('sphinx', 'foo')
        assert classes == []
        classes = import_classes('sphinx.parsers', None)
        assert set(classes) == {Parser, RSTParser}
        classes = import_classes('sphinx.parsers.Parser', None)
        assert classes == [Parser]
        classes = import_classes('Parser', 'sphinx.parsers')
        assert classes == [Parser]
        classes = import_classes('i18n.CatalogInfo', 'sphinx.util')
        assert classes == [CatalogInfo]
        with pytest.raises(InheritanceException):
            import_classes('encode_uri', 'sphinx.util')
        classes = import_classes('sphinx', 'example')
        assert classes == [DummyClass]
    finally:
        sys.path.pop()