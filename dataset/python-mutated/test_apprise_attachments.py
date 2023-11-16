import sys
import pytest
from os.path import getsize
from os.path import join
from os.path import dirname
from apprise.AppriseAttachment import AppriseAttachment
from apprise.AppriseAsset import AppriseAsset
from apprise.attachment.AttachBase import AttachBase
from apprise.common import ATTACHMENT_SCHEMA_MAP
from apprise.attachment import __load_matrix
from apprise.common import ContentLocation
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = join(dirname(__file__), 'var')

def test_apprise_attachment():
    if False:
        while True:
            i = 10
    '\n    API: AppriseAttachment basic testing\n\n    '
    aa = AppriseAttachment()
    assert len(aa) == 0
    assert not aa
    aa = AppriseAttachment(asset=AppriseAsset(), cache=300)
    assert len(aa) == 0
    path = join(TEST_VAR_DIR, 'apprise-test.gif')
    assert aa.add(path)
    assert len(aa) == 1
    assert aa[0].cache == 300
    assert aa
    response = AppriseAttachment.instantiate(path, cache=True)
    assert isinstance(response, AttachBase)
    assert aa.add(response, asset=AppriseAsset())
    assert len(aa) == 2
    assert aa[1].cache is True
    aa = AppriseAttachment()
    attachments = (path, 'file://{}?name=newfilename.gif?cache=120'.format(path), AppriseAttachment.instantiate('file://{}?name=anotherfilename.gif'.format(path), cache=100))
    assert aa.add(attachments, cache=False)
    assert len(aa) == 3
    assert aa[0].cache is False
    assert aa[1].cache is False
    assert aa[2].cache == 100
    attachment = aa.pop()
    assert isinstance(attachment, AttachBase)
    assert attachment
    assert len(aa) == 2
    assert attachment.path == path
    assert attachment.name == 'anotherfilename.gif'
    assert attachment.mimetype == 'image/gif'
    assert isinstance(aa[0], AttachBase)
    assert isinstance(aa[1], AttachBase)
    with pytest.raises(IndexError):
        aa[2]
    for (count, a) in enumerate(aa):
        assert isinstance(a, AttachBase)
        assert count < len(aa)
    expected_size = getsize(path) * len(aa)
    assert aa.size() == expected_size
    aa = AppriseAttachment(attachments)
    assert len(aa) == 3
    aa.clear()
    assert len(aa) == 0
    assert not aa
    assert aa.add(AppriseAttachment.instantiate('file://{}?name=andanother.png&cache=Yes'.format(path)))
    assert aa.add(AppriseAttachment.instantiate('file://{}?name=andanother.png&cache=No'.format(path)))
    AppriseAttachment.instantiate('file://{}?name=andanother.png&cache=600'.format(path))
    assert aa.add(AppriseAttachment.instantiate('file://{}?name=andanother.png&cache=600'.format(path)))
    assert len(aa) == 3
    assert aa[0].cache is True
    assert aa[1].cache is False
    assert aa[2].cache == 600
    assert not aa.add(AppriseAttachment.instantiate('file://{}?name=andanother.png&cache=-600'.format(path)))
    assert not aa.add(AppriseAttachment.instantiate('file://{}?name=andanother.png'.format(path), cache='invalid'))
    assert len(aa) == 3
    aa.clear()
    assert aa.add(None) is False
    assert aa.add(object()) is False
    assert aa.add(42) is False
    assert len(aa) == 0
    attachments = (None, object(), 42, 'garbage://')
    assert aa.add(attachments) is False
    assert len(aa) == 0
    with pytest.raises(TypeError):
        AppriseAttachment('garbage://')
    aa = AppriseAttachment(location=ContentLocation.LOCAL)
    aa = AppriseAttachment(location=ContentLocation.HOSTED)
    assert len(aa) == 0
    aa.add(attachments)
    assert len(aa) == 0
    aa = AppriseAttachment(location=ContentLocation.INACCESSIBLE)
    assert len(aa) == 0
    aa.add(attachments)
    assert len(aa) == 0
    with pytest.raises(TypeError):
        AppriseAttachment(location='invalid')
    aa = AppriseAttachment('file://non-existant-file.png')
    assert len(aa) == 1
    assert aa
    assert not aa[0]
    assert len(aa[0]) == 0
    assert aa.size() == 0

def test_apprise_attachment_instantiate():
    if False:
        i = 10
        return i + 15
    '\n    API: AppriseAttachment.instantiate()\n\n    '
    assert AppriseAttachment.instantiate('file://?', suppress_exceptions=True) is None
    assert AppriseAttachment.instantiate('invalid://?', suppress_exceptions=True) is None

    class BadAttachType(AttachBase):

        def __init__(self, **kwargs):
            if False:
                return 10
            super().__init__(**kwargs)
            raise TypeError()
    ATTACHMENT_SCHEMA_MAP['bad'] = BadAttachType
    with pytest.raises(TypeError):
        AppriseAttachment.instantiate('bad://path', suppress_exceptions=False)
    assert AppriseAttachment.instantiate('bad://path', suppress_exceptions=True) is None

def test_apprise_attachment_matrix_load():
    if False:
        for i in range(10):
            print('nop')
    '\n    API: AppriseAttachment() matrix initialization\n\n    '
    import apprise

    class AttachmentDummy(AttachBase):
        """
        A dummy wrapper for testing the different options in the load_matrix
        function
        """
        service_name = 'dummy'
        protocol = ('uh', 'oh')
        secure_protocol = ('no', 'yes')

    class AttachmentDummy2(AttachBase):
        """
        A dummy wrapper for testing the different options in the load_matrix
        function
        """
        service_name = 'dummy2'
        secure_protocol = ('true', 'false')

    class AttachmentDummy3(AttachBase):
        """
        A dummy wrapper for testing the different options in the load_matrix
        function
        """
        service_name = 'dummy3'
        secure_protocol = 'true'

    class AttachmentDummy4(AttachBase):
        """
        A dummy wrapper for testing the different options in the load_matrix
        function
        """
        service_name = 'dummy4'
        protocol = 'true'
    apprise.attachment.AttachmentDummy = AttachmentDummy
    apprise.attachment.AttachmentDummy2 = AttachmentDummy2
    apprise.attachment.AttachmentDummy3 = AttachmentDummy3
    apprise.attachment.AttachmentDummy4 = AttachmentDummy4
    __load_matrix()
    __load_matrix()

def test_attachment_matrix_dynamic_importing(tmpdir):
    if False:
        while True:
            i = 10
    '\n    API: Apprise() Attachment Matrix Importing\n\n    '
    suite = tmpdir.mkdir('apprise_attach_test_suite')
    suite.join('__init__.py').write('')
    module_name = 'badattach'
    sys.path.insert(0, str(suite))
    base = suite.mkdir(module_name)
    base.join('__init__.py').write('')
    base.join('AttachBadFile1.py').write('\nclass AttachBadFile1:\n    pass')
    base.join('AttachBadFile2.py').write('\nclass BadClassName:\n    pass')
    base.join('AttachBadFile3.py').write('raise ImportError()')
    base.join('AttachGoober.py').write("\nfrom apprise import AttachBase\nclass AttachGoober(AttachBase):\n    # This class tests the fact we have a new class name, but we're\n    # trying to over-ride items previously used\n\n    # The default simple (insecure) protocol\n    protocol = 'http'\n\n    # The default secure protocol\n    secure_protocol = 'https'")
    base.join('AttachBugger.py').write("\nfrom apprise import AttachBase\nclass AttachBugger(AttachBase):\n    # This class tests the fact we have a new class name, but we're\n    # trying to over-ride items previously used\n\n    # The default simple (insecure) protocol\n    protocol = ('http', 'bugger-test' )\n\n    # The default secure protocol\n    secure_protocol = ('https', 'bugger-tests')")
    __load_matrix(path=str(base), name=module_name)