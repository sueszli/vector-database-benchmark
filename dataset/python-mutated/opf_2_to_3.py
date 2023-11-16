from lxml import etree
from calibre.ebooks.metadata.opf3 import DC, OPF, XPath, create_rating, create_series, create_timestamp, encode_is_multiple, ensure_id, normalize_whitespace, parse_date, read_prefixes, read_refines, read_user_metadata2, refdef, remove_element, set_last_modified, set_refines, set_user_metadata3
from calibre.ebooks.metadata.utils import parse_opf, pretty_print_opf
from polyglot.builtins import itervalues

class Data:
    pass

def upgrade_identifiers(root, data):
    if False:
        for i in range(10):
            print('nop')
    for ident in XPath('./opf:metadata/dc:identifier')(root):
        val = (ident.text or '').strip()
        lval = val.lower()
        scheme = ident.attrib.pop(OPF('scheme'), None)
        if lval.startswith('urn:'):
            (prefix, rest) = val[4:].partition(':')[::2]
            if prefix and rest:
                (scheme, val) = (prefix, rest)
        if scheme and val:
            ident.text = f'{scheme}:{val}'
        for attr in tuple(ident.attrib):
            if attr != 'id':
                del ident.attrib[attr]

def upgrade_title(root, data):
    if False:
        for i in range(10):
            print('nop')
    first_title = None
    for title in XPath('./opf:metadata/dc:title')(root):
        if not title.text or not title.text.strip():
            remove_element(title, data.refines)
            continue
        if first_title is None:
            first_title = title
    title_sort = None
    for m in XPath('./opf:metadata/opf:meta[@name="calibre:title_sort"]')(root):
        ans = m.get('content')
        if ans:
            title_sort = ans
        remove_element(m, data.refines)
    if first_title is not None:
        ts = [refdef('file-as', title_sort)] if title_sort else ()
        set_refines(first_title, data.refines, refdef('title-type', 'main'), *ts)

def upgrade_languages(root, data):
    if False:
        while True:
            i = 10
    langs = XPath('./opf:metadata/dc:language')(root)
    if langs:
        for lang in langs:
            lang.attrib.clear()
    else:
        metadata = XPath('./opf:metadata')(root)[0]
        l = metadata.makeelement(DC('language'))
        l.text = 'und'
        metadata.append(l)

def upgrade_authors(root, data):
    if False:
        print('Hello World!')
    for which in ('creator', 'contributor'):
        for elem in XPath('./opf:metadata/dc:' + which)(root):
            role = elem.attrib.pop(OPF('role'), None)
            sort = elem.attrib.pop(OPF('file-as'), None)
            if role or sort:
                aid = ensure_id(elem)
                metadata = elem.getparent()
                if role:
                    m = metadata.makeelement(OPF('meta'), attrib={'refines': '#' + aid, 'property': 'role', 'scheme': 'marc:relators'})
                    m.text = role
                    metadata.append(m)
                if sort:
                    m = metadata.makeelement(OPF('meta'), attrib={'refines': '#' + aid, 'property': 'file-as'})
                    m.text = sort
                    metadata.append(m)

def upgrade_timestamp(root, data):
    if False:
        for i in range(10):
            print('nop')
    for meta in XPath('./opf:metadata/opf:meta[@name="calibre:timestamp"]')(root):
        m = meta.getparent()
        remove_element(meta, data.refines)
        val = meta.get('content')
        if val:
            try:
                val = parse_date(val, is_w3cdtf=True)
            except Exception:
                pass
            else:
                create_timestamp(root, data.prefixes, m, val)

def upgrade_date(root, data):
    if False:
        print('Hello World!')
    found = False
    for date in XPath('./opf:metadata/dc:date')(root):
        val = date.text
        if not val:
            remove_element(date, data.refines)
            continue
        if found:
            remove_element(date, data.refines)
        else:
            found = True

def upgrade_rating(root, data):
    if False:
        for i in range(10):
            print('nop')
    rating = None
    for meta in XPath('./opf:metadata/opf:meta[@name="calibre:rating"]')(root):
        remove_element(meta, data.refines)
        rating = meta.get('content')
    if rating is not None:
        create_rating(root, data.prefixes, rating)

def upgrade_series(root, data):
    if False:
        return 10
    (series, series_index) = (None, '1.0')
    for meta in XPath('./opf:metadata/opf:meta[@name="calibre:series"]')(root):
        remove_element(meta, data.refines)
        series = meta.get('content')
    for meta in XPath('./opf:metadata/opf:meta[@name="calibre:series_index"]')(root):
        remove_element(meta, data.refines)
        series_index = meta.get('content')
    if series:
        create_series(root, data.refines, series, series_index)

def upgrade_custom(root, data):
    if False:
        while True:
            i = 10
    m = read_user_metadata2(root, remove_tags=True)
    if m:
        for fm in itervalues(m):
            encode_is_multiple(fm)
        set_user_metadata3(root, data.prefixes, data.refines, m)

def upgrade_meta(root, data):
    if False:
        for i in range(10):
            print('nop')
    for meta in XPath('./opf:metadata/opf:meta[@name]')(root):
        (name, content) = (meta.get('name'), meta.get('content') or '')
        if name.startswith('rendition:'):
            name = name.partition(':')[-1]
        prop = None
        if name in ('orientation', 'layout', 'spread'):
            prop = 'rendition:' + name
        elif name == 'fixed-layout':
            prop = 'rendition:layout'
            content = {'true': 'pre-paginated'}.get(content.lower(), 'reflowable')
        elif name == 'orientation-lock':
            prop = 'rendition:orientation'
            content = {'portrait': 'portrait', 'landscape': 'landscape'}.get(content.lower(), 'auto')
        if prop:
            del meta.attrib['name']
            del meta.attrib['content']
            meta.set('property', prop)
            meta.text = content

def upgrade_cover(root, data):
    if False:
        print('Hello World!')
    for item in XPath('./opf:metadata/opf:meta[@name="cover"]')(root):
        item_id = item.get('content')
        for item in XPath('./opf:manifest/opf:item[@id and @href and @media-type]')(root):
            if item.get('id') == item_id:
                mt = (item.get('media-type') or '').lower()
                if mt and 'xml' not in mt and ('html' not in mt):
                    item.set('properties', normalize_whitespace((item.get('properties') or '') + ' cover-image'))

def remove_invalid_attrs_in_dc_metadata(root, data):
    if False:
        i = 10
        return i + 15
    for tag in XPath('//*[namespace-uri() = "{}"]'.format(DC('')[1:-1]))(root):
        for k in tuple(tag.attrib):
            if k != 'id':
                del tag.attrib[k]

def upgrade_metadata(root):
    if False:
        while True:
            i = 10
    data = Data()
    data.prefixes = read_prefixes(root)
    data.refines = read_refines(root)
    upgrade_identifiers(root, data)
    upgrade_title(root, data)
    upgrade_languages(root, data)
    upgrade_authors(root, data)
    upgrade_timestamp(root, data)
    upgrade_date(root, data)
    upgrade_rating(root, data)
    upgrade_series(root, data)
    upgrade_custom(root, data)
    upgrade_meta(root, data)
    upgrade_cover(root, data)
    remove_invalid_attrs_in_dc_metadata(root, data)
    set_last_modified(root, data.prefixes, data.refines)
    pretty_print_opf(root)
if __name__ == '__main__':
    import sys
    root = parse_opf(open(sys.argv[-1], 'rb'))
    upgrade_metadata(root)
    print(etree.tostring(root))