from bzrlib.chk_serializer import chk_bencode_serializer
from bzrlib.revision import Revision
from bzrlib.tests import TestCase
_working_revision_bencode1 = 'll6:formati10eel9:committer54:Canonical.com Patch Queue Manager <pqm@pqm.ubuntu.com>el8:timezonei3600eel10:propertiesd11:branch-nick6:+trunkeel9:timestamp14:1242300770.844el11:revision-id50:pqm@pqm.ubuntu.com-20090514113250-jntkkpminfn3e0tzel10:parent-idsl50:pqm@pqm.ubuntu.com-20090514104039-kggemn7lrretzpvc48:jelmer@samba.org-20090510012654-jp9ufxquekaokbeoeel14:inventory-sha140:4a2c7fb50e077699242cf6eb16a61779c7b680a7el7:message35:(Jelmer) Move dpush to InterBranch.ee'
_working_revision_bencode1_no_timezone = 'll6:formati10eel9:committer54:Canonical.com Patch Queue Manager <pqm@pqm.ubuntu.com>el9:timestamp14:1242300770.844el10:propertiesd11:branch-nick6:+trunkeel11:revision-id50:pqm@pqm.ubuntu.com-20090514113250-jntkkpminfn3e0tzel10:parent-idsl50:pqm@pqm.ubuntu.com-20090514104039-kggemn7lrretzpvc48:jelmer@samba.org-20090510012654-jp9ufxquekaokbeoeel14:inventory-sha140:4a2c7fb50e077699242cf6eb16a61779c7b680a7el7:message35:(Jelmer) Move dpush to InterBranch.ee'

class TestBEncodeSerializer1(TestCase):
    """Test BEncode serialization"""

    def test_unpack_revision(self):
        if False:
            print('Hello World!')
        'Test unpacking a revision'
        rev = chk_bencode_serializer.read_revision_from_string(_working_revision_bencode1)
        self.assertEqual(rev.committer, 'Canonical.com Patch Queue Manager <pqm@pqm.ubuntu.com>')
        self.assertEqual(rev.inventory_sha1, '4a2c7fb50e077699242cf6eb16a61779c7b680a7')
        self.assertEqual(['pqm@pqm.ubuntu.com-20090514104039-kggemn7lrretzpvc', 'jelmer@samba.org-20090510012654-jp9ufxquekaokbeo'], rev.parent_ids)
        self.assertEqual('(Jelmer) Move dpush to InterBranch.', rev.message)
        self.assertEqual('pqm@pqm.ubuntu.com-20090514113250-jntkkpminfn3e0tz', rev.revision_id)
        self.assertEqual({'branch-nick': u'+trunk'}, rev.properties)
        self.assertEqual(3600, rev.timezone)

    def test_written_form_matches(self):
        if False:
            for i in range(10):
                print('nop')
        rev = chk_bencode_serializer.read_revision_from_string(_working_revision_bencode1)
        as_str = chk_bencode_serializer.write_revision_to_string(rev)
        self.assertEqualDiff(_working_revision_bencode1, as_str)

    def test_unpack_revision_no_timezone(self):
        if False:
            print('Hello World!')
        rev = chk_bencode_serializer.read_revision_from_string(_working_revision_bencode1_no_timezone)
        self.assertEqual(None, rev.timezone)

    def assertRoundTrips(self, serializer, orig_rev):
        if False:
            while True:
                i = 10
        text = serializer.write_revision_to_string(orig_rev)
        new_rev = serializer.read_revision_from_string(text)
        self.assertEqual(orig_rev, new_rev)

    def test_roundtrips_non_ascii(self):
        if False:
            print('Hello World!')
        rev = Revision('revid1')
        rev.message = u'\nåme'
        rev.committer = u'Erik Bågfors'
        rev.timestamp = 1242385452
        rev.inventory_sha1 = '4a2c7fb50e077699242cf6eb16a61779c7b680a7'
        rev.timezone = 3600
        self.assertRoundTrips(chk_bencode_serializer, rev)

    def test_roundtrips_xml_invalid_chars(self):
        if False:
            for i in range(10):
                print('nop')
        rev = Revision('revid1')
        rev.message = '\t\ue000'
        rev.committer = u'Erik Bågfors'
        rev.timestamp = 1242385452
        rev.timezone = 3600
        rev.inventory_sha1 = '4a2c7fb50e077699242cf6eb16a61779c7b680a7'
        self.assertRoundTrips(chk_bencode_serializer, rev)