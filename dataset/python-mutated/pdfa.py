"""PDF/A generation."""
try:
    from importlib.resources import files
except ImportError:
    from importlib.resources import read_binary
else:

    def read_binary(package, resource):
        if False:
            i = 10
            return i + 15
        return (files(package) / resource).read_bytes()
from functools import partial
import pydyf
from ..logger import LOGGER
from .metadata import add_metadata

def pdfa(pdf, metadata, document, page_streams, compress, version):
    if False:
        while True:
            i = 10
    'Set metadata for PDF/A documents.'
    LOGGER.warning('PDF/A support is experimental, generated PDF files are not guaranteed to be valid. Please open an issue if you have problems generating PDF/A files.')
    profile = pydyf.Stream([read_binary(__package__, 'sRGB2014.icc')], pydyf.Dictionary({'N': 3, 'Alternate': '/DeviceRGB'}), compress=compress)
    pdf.add_object(profile)
    pdf.catalog['OutputIntents'] = pydyf.Array([pydyf.Dictionary({'Type': '/OutputIntent', 'S': '/GTS_PDFA1', 'OutputConditionIdentifier': pydyf.String('sRGB IEC61966-2.1'), 'DestOutputProfile': profile.reference})])
    for pdf_object in pdf.objects:
        if isinstance(pdf_object, dict) and pdf_object.get('Type') == '/Annot':
            pdf_object['F'] = 2 ** (3 - 1)
    add_metadata(pdf, metadata, 'a', version, 'B', compress)
VARIANTS = {f'pdf/a-{i}b': (partial(pdfa, version=i), {'version': pdf_version}) for (i, pdf_version) in enumerate(('1.4', '1.7', '1.7', '2.0'), start=1)}