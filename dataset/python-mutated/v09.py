from __future__ import absolute_import
from bzrlib.bundle.serializer import _get_bundle_header
from bzrlib.bundle.serializer.v08 import BundleSerializerV08, BundleReader
from bzrlib.testament import StrictTestament3
from bzrlib.bundle.bundle_data import BundleInfo
'Serializer for bundle format 0.9'

class BundleSerializerV09(BundleSerializerV08):
    """Serializer for bzr bundle format 0.9

    This format supports rich root data, for the nested-trees work, but also
    supports repositories that don't have rich root data.  It cannot be
    used to transfer from a knit2 repo into a knit1 repo, because that would
    be lossy.
    """

    def check_compatible(self):
        if False:
            return 10
        pass

    def _write_main_header(self):
        if False:
            i = 10
            return i + 15
        'Write the header for the changes'
        f = self.to_file
        f.write(_get_bundle_header('0.9') + '#\n')

    def _testament_sha1(self, revision_id):
        if False:
            i = 10
            return i + 15
        return StrictTestament3.from_revision(self.source, revision_id).as_sha1()

    def read(self, f):
        if False:
            print('Hello World!')
        'Read the rest of the bundles from the supplied file.\n\n        :param f: The file to read from\n        :return: A list of bundles\n        '
        return BundleReaderV09(f).info

class BundleInfo09(BundleInfo):
    """BundleInfo that uses StrictTestament3

    This means that the root data is included in the testament.
    """

    def _testament_sha1_from_revision(self, repository, revision_id):
        if False:
            while True:
                i = 10
        testament = StrictTestament3.from_revision(repository, revision_id)
        return testament.as_sha1()

    def _testament_sha1(self, revision, tree):
        if False:
            for i in range(10):
                print('nop')
        return StrictTestament3(revision, tree).as_sha1()

class BundleReaderV09(BundleReader):
    """BundleReader for 0.9 bundles"""

    def _get_info(self):
        if False:
            for i in range(10):
                print('nop')
        return BundleInfo09()