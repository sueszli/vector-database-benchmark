__license__ = 'GPL v3'
__copyright__ = '2011, John Schember <john@nachtimwald.com>'
'\nRead meta information from extZ (TXTZ, HTMLZ...) files.\n'
import io
import os
from calibre.ebooks.metadata import MetaInformation
from calibre.ebooks.metadata.opf2 import OPF
from calibre.ptempfile import PersistentTemporaryFile
from calibre.utils.zipfile import ZipFile, safe_replace

def get_metadata(stream, extract_cover=True):
    if False:
        while True:
            i = 10
    '\n    Return metadata as a L{MetaInfo} object\n    '
    mi = MetaInformation(_('Unknown'), [_('Unknown')])
    stream.seek(0)
    try:
        with ZipFile(stream) as zf:
            opf_name = get_first_opf_name(zf)
            with zf.open(opf_name) as opf_stream:
                opf = OPF(opf_stream)
            mi = opf.to_book_metadata()
            if extract_cover:
                cover_href = opf.raster_cover or opf.guide_raster_cover
                if not cover_href:
                    for meta in opf.metadata.xpath('//*[local-name()="meta" and @name="cover"]'):
                        val = meta.get('content')
                        if val.rpartition('.')[2].lower() in {'jpeg', 'jpg', 'png'}:
                            cover_href = val
                            break
                    else:
                        for val in opf.guide_cover_path(opf.root):
                            if val.rpartition('.')[2].lower() in {'jpeg', 'jpg', 'png'}:
                                cover_href = val
                                break
                        else:
                            for cpath in opf.root.xpath('//cover-relpath-from-base'):
                                if cpath.text:
                                    cover_href = cpath.text
                                    break
                if cover_href:
                    try:
                        mi.cover_data = (os.path.splitext(cover_href)[1], zf.read(cover_href))
                    except Exception:
                        pass
    except Exception:
        return mi
    return mi

def set_metadata(stream, mi):
    if False:
        print('Hello World!')
    replacements = {}
    with ZipFile(stream) as zf:
        opf_path = get_first_opf_name(zf)
        opf_stream = io.BytesIO(zf.read(opf_path))
    opf = OPF(opf_stream)
    new_cdata = None
    try:
        new_cdata = mi.cover_data[1]
        if not new_cdata:
            raise Exception('no cover')
    except:
        try:
            with open(mi.cover, 'rb') as f:
                new_cdata = f.read()
        except:
            pass
    if new_cdata:
        cpath = opf.raster_cover
        if not cpath:
            cpath = 'cover.jpg'
        new_cover = _write_new_cover(new_cdata, cpath)
        replacements[cpath] = open(new_cover.name, 'rb')
        mi.cover = cpath
    opf.smart_update(mi, replace_metadata=True)
    newopf = io.BytesIO(opf.render())
    safe_replace(stream, opf_path, newopf, extra_replacements=replacements, add_missing=True)
    try:
        if cpath is not None:
            replacements[cpath].close()
            os.remove(replacements[cpath].name)
    except:
        pass

def get_first_opf_name(zf):
    if False:
        for i in range(10):
            print('nop')
    names = zf.namelist()
    opfs = []
    for n in names:
        if n.endswith('.opf') and '/' not in n:
            opfs.append(n)
    if not opfs:
        raise Exception('No OPF found')
    opfs.sort()
    return opfs[0]

def _write_new_cover(new_cdata, cpath):
    if False:
        i = 10
        return i + 15
    from calibre.utils.img import save_cover_data_to
    new_cover = PersistentTemporaryFile(suffix=os.path.splitext(cpath)[1])
    new_cover.close()
    save_cover_data_to(new_cdata, new_cover.name)
    return new_cover