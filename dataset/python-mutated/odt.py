import io
import json
import os
import re
from lxml.etree import fromstring, tostring
from calibre.ebooks.metadata import MetaInformation, authors_to_string, check_isbn, string_to_authors
from calibre.utils.date import isoformat, parse_date
from calibre.utils.imghdr import identify
from calibre.utils.localization import canonicalize_lang, lang_as_iso639_1
from calibre.utils.zipfile import ZipFile, safe_replace
from odf.draw import Frame as odFrame, Image as odImage
from odf.namespaces import DCNS, METANS, OFFICENS
from odf.opendocument import load as odLoad
from polyglot.builtins import as_unicode
fields = {'title': (DCNS, 'title'), 'description': (DCNS, 'description'), 'subject': (DCNS, 'subject'), 'creator': (DCNS, 'creator'), 'date': (DCNS, 'date'), 'language': (DCNS, 'language'), 'generator': (METANS, 'generator'), 'initial-creator': (METANS, 'initial-creator'), 'keyword': (METANS, 'keyword'), 'keywords': (METANS, 'keywords'), 'editing-duration': (METANS, 'editing-duration'), 'editing-cycles': (METANS, 'editing-cycles'), 'printed-by': (METANS, 'printed-by'), 'print-date': (METANS, 'print-date'), 'creation-date': (METANS, 'creation-date'), 'user-defined': (METANS, 'user-defined')}

def uniq(vals):
    if False:
        while True:
            i = 10
    ' Remove all duplicates from vals, while preserving order.  '
    vals = vals or ()
    seen = set()
    seen_add = seen.add
    return list((x for x in vals if x not in seen and (not seen_add(x))))

def get_metadata(stream, extract_cover=True):
    if False:
        i = 10
        return i + 15
    whitespace = re.compile('\\s+')

    def normalize(s):
        if False:
            for i in range(10):
                print('nop')
        return whitespace.sub(' ', s).strip()
    with ZipFile(stream) as zf:
        meta = zf.read('meta.xml')
        root = fromstring(meta)

        def find(field):
            if False:
                print('Hello World!')
            (ns, tag) = fields[field]
            ans = root.xpath(f'//ns0:{tag}', namespaces={'ns0': ns})
            if ans:
                return normalize(tostring(ans[0], method='text', encoding='unicode', with_tail=False)).strip()

        def find_all(field):
            if False:
                for i in range(10):
                    print('nop')
            (ns, tag) = fields[field]
            for x in root.xpath(f'//ns0:{tag}', namespaces={'ns0': ns}):
                yield normalize(tostring(x, method='text', encoding='unicode', with_tail=False)).strip()
        mi = MetaInformation(None, [])
        title = find('title')
        if title:
            mi.title = title
        creator = find('initial-creator') or find('creator')
        if creator:
            mi.authors = string_to_authors(creator)
        desc = find('description')
        if desc:
            mi.comments = desc
        lang = find('language')
        if lang and canonicalize_lang(lang):
            mi.languages = [canonicalize_lang(lang)]
        keywords = []
        for q in ('keyword', 'keywords'):
            for kw in find_all(q):
                keywords += [x.strip() for x in kw.split(',') if x.strip()]
        mi.tags = uniq(keywords)
        data = {}
        for tag in root.xpath('//ns0:user-defined', namespaces={'ns0': fields['user-defined'][0]}):
            name = (tag.get('{%s}name' % METANS) or '').lower()
            vtype = tag.get('{%s}value-type' % METANS) or 'string'
            val = tag.text
            if name and val:
                if vtype == 'boolean':
                    val = val == 'true'
                data[name] = val
        opfmeta = False
        opfnocover = False
        if data.get('opf.metadata'):
            opfmeta = True
            if data.get('opf.titlesort', ''):
                mi.title_sort = data['opf.titlesort']
            if data.get('opf.authors', ''):
                mi.authors = string_to_authors(data['opf.authors'])
            if data.get('opf.authorsort', ''):
                mi.author_sort = data['opf.authorsort']
            if data.get('opf.isbn', ''):
                isbn = check_isbn(data['opf.isbn'])
                if isbn is not None:
                    mi.isbn = isbn
            if data.get('opf.publisher', ''):
                mi.publisher = data['opf.publisher']
            if data.get('opf.pubdate', ''):
                mi.pubdate = parse_date(data['opf.pubdate'], assume_utc=True)
            if data.get('opf.identifiers'):
                try:
                    mi.identifiers = json.loads(data['opf.identifiers'])
                except Exception:
                    pass
            if data.get('opf.rating'):
                try:
                    mi.rating = max(0, min(float(data['opf.rating']), 10))
                except Exception:
                    pass
            if data.get('opf.series', ''):
                mi.series = data['opf.series']
                if data.get('opf.seriesindex', ''):
                    try:
                        mi.series_index = float(data['opf.seriesindex'])
                    except Exception:
                        mi.series_index = 1.0
            if data.get('opf.language', ''):
                cl = canonicalize_lang(data['opf.language'])
                if cl:
                    mi.languages = [cl]
            opfnocover = data.get('opf.nocover', False)
        if not opfnocover:
            try:
                read_cover(stream, zf, mi, opfmeta, extract_cover)
            except Exception:
                pass
    return mi

def set_metadata(stream, mi):
    if False:
        for i in range(10):
            print('nop')
    with ZipFile(stream) as zf:
        raw = _set_metadata(zf.open('meta.xml').read(), mi)
    stream.seek(os.SEEK_SET)
    safe_replace(stream, 'meta.xml', io.BytesIO(raw))

def _set_metadata(raw, mi):
    if False:
        i = 10
        return i + 15
    root = fromstring(raw)
    namespaces = {'office': OFFICENS, 'meta': METANS, 'dc': DCNS}
    nsrmap = {v: k for (k, v) in namespaces.items()}

    def xpath(expr, parent=root):
        if False:
            for i in range(10):
                print('nop')
        return parent.xpath(expr, namespaces=namespaces)

    def remove(*tag_names):
        if False:
            while True:
                i = 10
        for tag_name in tag_names:
            ns = fields[tag_name][0]
            tag_name = f'{nsrmap[ns]}:{tag_name}'
            for x in xpath('descendant::' + tag_name, meta):
                x.getparent().remove(x)

    def add(tag, val=None):
        if False:
            print('Hello World!')
        ans = meta.makeelement('{%s}%s' % fields[tag])
        ans.text = val
        meta.append(ans)
        return ans

    def remove_user_metadata(*names):
        if False:
            for i in range(10):
                print('nop')
        for x in xpath('//meta:user-defined'):
            q = (x.get('{%s}name' % METANS) or '').lower()
            if q in names:
                x.getparent().remove(x)

    def add_um(name, val, vtype='string'):
        if False:
            for i in range(10):
                print('nop')
        ans = add('user-defined', val)
        ans.set('{%s}value-type' % METANS, vtype)
        ans.set('{%s}name' % METANS, name)

    def add_user_metadata(name, val):
        if False:
            while True:
                i = 10
        if not hasattr(add_user_metadata, 'sentinel_added'):
            add_user_metadata.sentinel_added = True
            remove_user_metadata('opf.metadata')
            add_um('opf.metadata', 'true', 'boolean')
        val_type = 'string'
        if hasattr(val, 'strftime'):
            val = isoformat(val, as_utc=True).split('T')[0]
            val_type = 'date'
        add_um(name, val, val_type)
    meta = xpath('//office:meta')[0]
    if not mi.is_null('title'):
        remove('title')
        add('title', mi.title)
        if not mi.is_null('title_sort'):
            remove_user_metadata('opf.titlesort')
            add_user_metadata('opf.titlesort', mi.title_sort)
    if not mi.is_null('authors'):
        remove('initial-creator', 'creator')
        val = authors_to_string(mi.authors)
        (add('initial-creator', val), add('creator', val))
        remove_user_metadata('opf.authors')
        add_user_metadata('opf.authors', val)
        if not mi.is_null('author_sort'):
            remove_user_metadata('opf.authorsort')
            add_user_metadata('opf.authorsort', mi.author_sort)
    if not mi.is_null('comments'):
        remove('description')
        add('description', mi.comments)
    if not mi.is_null('tags'):
        remove('keyword')
        add('keyword', ', '.join(mi.tags))
    if not mi.is_null('languages'):
        lang = lang_as_iso639_1(mi.languages[0])
        if lang:
            remove('language')
            add('language', lang)
    if not mi.is_null('pubdate'):
        remove_user_metadata('opf.pubdate')
        add_user_metadata('opf.pubdate', mi.pubdate)
    if not mi.is_null('publisher'):
        remove_user_metadata('opf.publisher')
        add_user_metadata('opf.publisher', mi.publisher)
    if not mi.is_null('series'):
        remove_user_metadata('opf.series', 'opf.seriesindex')
        add_user_metadata('opf.series', mi.series)
        add_user_metadata('opf.seriesindex', f'{mi.series_index}')
    if not mi.is_null('identifiers'):
        remove_user_metadata('opf.identifiers')
        add_user_metadata('opf.identifiers', as_unicode(json.dumps(mi.identifiers)))
    if not mi.is_null('rating'):
        remove_user_metadata('opf.rating')
        add_user_metadata('opf.rating', '%.2g' % mi.rating)
    return tostring(root, encoding='utf-8', pretty_print=True)

def read_cover(stream, zin, mi, opfmeta, extract_cover):
    if False:
        return 10
    otext = odLoad(stream)
    cover_href = None
    cover_data = None
    cover_frame = None
    imgnum = 0
    for frm in otext.topnode.getElementsByType(odFrame):
        img = frm.getElementsByType(odImage)
        if len(img) == 0:
            continue
        i_href = img[0].getAttribute('href')
        try:
            raw = zin.read(i_href)
        except KeyError:
            continue
        try:
            (fmt, width, height) = identify(raw)
        except Exception:
            continue
        imgnum += 1
        if opfmeta and frm.getAttribute('name').lower() == 'opf.cover':
            cover_href = i_href
            cover_data = (fmt, raw)
            cover_frame = frm.getAttribute('name')
            break
        if cover_href is None and imgnum == 1 and (0.8 <= height / width <= 1.8) and (height * width >= 12000):
            cover_href = i_href
            cover_data = (fmt, raw)
            if not opfmeta:
                break
    if cover_href is not None:
        mi.cover = cover_href
        mi.odf_cover_frame = cover_frame
        if extract_cover:
            if not cover_data:
                raw = zin.read(cover_href)
                try:
                    fmt = identify(raw)[0]
                except Exception:
                    pass
                else:
                    cover_data = (fmt, raw)
            mi.cover_data = cover_data