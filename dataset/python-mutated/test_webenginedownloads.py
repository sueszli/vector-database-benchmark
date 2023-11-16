import base64
import dataclasses
import pytest
pytest.importorskip('qutebrowser.qt.webenginecore')
from qutebrowser.qt.webenginecore import QWebEngineProfile
from qutebrowser.utils import urlutils, usertypes, utils
from qutebrowser.browser.webengine import webenginedownloads

@pytest.mark.parametrize('path, expected', [('foo(1)', 'foo'), ('foo (1)', 'foo'), ('foo - 1970-01-01T00:00:00.000Z', 'foo'), ('foo(a)', 'foo(a)'), ('foo1', 'foo1'), ('foo%20bar', 'foo%20bar'), ('foo%2Fbar', 'foo%2Fbar')])
def test_strip_suffix(path, expected):
    if False:
        while True:
            i = 10
    assert webenginedownloads._strip_suffix(path) == expected

@dataclasses.dataclass
class _ExpectedNames:
    """The filenames used in the tests."""
    before: str
    after: str

class TestDataUrlWorkaround:
    """With data URLs, we get rather weird base64 filenames back from QtWebEngine.

    See https://bugreports.qt.io/browse/QTBUG-90355
    """

    @pytest.fixture(params=[True, False])
    def pdf_bytes(self, request):
        if False:
            i = 10
            return i + 15
        with_slash = request.param
        pdf_source = ['%PDF-1.0', '1 0 obj<</Pages 2 0 R>>endobj', '2 0 obj<</Kids[3 0 R]/Count 1>>endobj', '3 0 obj<</MediaBox[0 0 3 3]>>endobj', 'trailer<</Root 1 0 R>>']
        if with_slash:
            pdf_source.insert(1, '% ?')
        return '\n'.join(pdf_source).encode('ascii')

    @pytest.fixture
    def pdf_url(self, pdf_bytes):
        if False:
            while True:
                i = 10
        return urlutils.data_url('application/pdf', pdf_bytes)

    @pytest.fixture
    def expected_names(self, webengine_versions, pdf_bytes):
        if False:
            for i in range(10):
                print('nop')
        'Get the expected filenames before/after the workaround.\n\n        With QtWebEngine 5.15.3, this is handled correctly inside QtWebEngine\n        and we get a qwe_download.pdf instead.\n        '
        if webengine_versions.webengine >= utils.VersionNumber(5, 15, 3):
            return _ExpectedNames(before='qwe_download.pdf', after='qwe_download.pdf')
        with_slash = b'% ?' in pdf_bytes
        base64_data = base64.b64encode(pdf_bytes).decode('ascii')
        if with_slash:
            assert '/' in base64_data
            before = base64_data.split('/')[1]
        else:
            assert '/' not in base64_data
            before = 'pdf'
        return _ExpectedNames(before=before, after='download.pdf')

    @pytest.fixture
    def webengine_profile(self, qapp):
        if False:
            for i in range(10):
                print('nop')
        profile = QWebEngineProfile.defaultProfile()
        profile.setParent(qapp)
        return profile

    @pytest.fixture
    def download_manager(self, qapp, qtbot, webengine_profile, download_tmpdir, config_stub):
        if False:
            while True:
                i = 10
        config_stub.val.downloads.location.suggestion = 'filename'
        manager = webenginedownloads.DownloadManager(parent=qapp)
        manager.install(webengine_profile)
        yield manager
        webengine_profile.downloadRequested.disconnect()

    def test_workaround(self, webengine_tab, message_mock, qtbot, pdf_url, download_manager, expected_names):
        if False:
            i = 10
            return i + 15
        'Verify our workaround works properly.'
        with qtbot.wait_signal(message_mock.got_question):
            webengine_tab.load_url(pdf_url)
        question = message_mock.get_question()
        assert question.default == expected_names.after

    def test_explicit_filename(self, webengine_tab, message_mock, qtbot, pdf_url, download_manager):
        if False:
            for i in range(10):
                print('nop')
        'If a website sets an explicit filename, we should respect that.'
        pdf_url_str = pdf_url.toDisplayString()
        html = f'<a href="{pdf_url_str}" download="filename.pdf" id="link">'
        with qtbot.wait_signal(webengine_tab.load_finished):
            webengine_tab.set_html(html)
        with qtbot.wait_signal(message_mock.got_question):
            webengine_tab.elements.find_id('link', lambda elem: elem.click(usertypes.ClickTarget.normal))
        question = message_mock.get_question()
        assert question.default == 'filename.pdf'

    def test_workaround_needed(self, qtbot, webengineview, pdf_url, expected_names, webengine_profile):
        if False:
            return 10
        'Verify that our workaround for this is still needed.\n\n        In other words, check whether we get those base64-filenames rather than a\n        "download.pdf" like with Chromium.\n        '

        def check_item(item):
            if False:
                i = 10
                return i + 15
            assert item.mimeType() == 'application/pdf'
            assert item.url().scheme() == 'data'
            assert item.downloadFileName() == expected_names.before
            return True
        with qtbot.wait_signal(webengine_profile.downloadRequested, check_params_cb=check_item):
            webengineview.load(pdf_url)